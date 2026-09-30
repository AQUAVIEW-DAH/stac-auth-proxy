"""Middleware to validate transaction requests against a CQL2 filter."""

import json
import re
from dataclasses import dataclass, field
from logging import getLogger
from typing import Optional
from urllib.parse import quote

import httpx
from cql2 import Expr
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ..utils.middleware import required_conformance

logger = getLogger(__name__)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Non-dict values from override replace base."""
    result = {**base}
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@required_conformance(
    r"http://www.opengis.net/spec/cql2/1.0/conf/basic-cql2",
    r"http://www.opengis.net/spec/cql2/1.0/conf/cql2-text",
    r"http://www.opengis.net/spec/cql2/1.0/conf/cql2-json",
)
@dataclass
class Cql2ValidateTransactionMiddleware:
    """Middleware to validate transaction requests against a CQL2 filter."""

    app: ASGIApp
    upstream_url: str
    state_key: str = "cql2_filter"
    read_state_key: str = "cql2_read_filter"

    _client: httpx.AsyncClient = field(init=False)

    # Transaction endpoint patterns
    items_pattern = r"^/collections/([^/]+)/(items|bulk_items)(?:/([^/]+))?$"
    collections_pattern = r"^/collections(?:/([^/]+))?$"
    # Multi-Tenant Catalogs endpoints, checked when a filter covers them
    # https://github.com/StacLabs/multi-tenant-catalogs/blob/v1.0.0/README.md#transactions-management
    catalogs_pattern = r"^/catalogs(?:/([^/]+))?$"
    catalog_children_pattern = (
        r"^/catalogs/([^/]+)/(catalogs|collections)(?:/([^/]+))?$"
    )

    def __post_init__(self):
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(base_url=self.upstream_url)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Validate transaction requests against the CQL2 filter."""
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request = Request(scope)
        cql2_filter: Optional[Expr] = getattr(request.state, self.state_key, None)
        if not cql2_filter:
            return await self.app(scope, receive, send)

        path = request.url.path
        method = request.method

        # Match items endpoints: /collections/{id}/items, /collections/{id}/bulk_items, /collections/{id}/items/{id}
        if re.match(self.items_pattern, path):
            if method == "POST":
                if "/bulk_items" in path:
                    return await self._handle_bulk_create(
                        scope, receive, send, cql2_filter
                    )
                return await self._handle_create(scope, receive, send, cql2_filter)
            if method in ("PUT", "PATCH"):
                return await self._handle_update(
                    scope, receive, send, cql2_filter, path, method
                )
            if method == "DELETE":
                return await self._handle_delete(
                    scope, receive, send, cql2_filter, path
                )

        # Match collections endpoints: /collections, /collections/{id}
        # and catalogs endpoints: /catalogs, /catalogs/{id}
        if re.match(self.collections_pattern, path) or re.match(
            self.catalogs_pattern, path
        ):
            if method == "POST":
                return await self._handle_create(scope, receive, send, cql2_filter)
            if method in ("PUT", "PATCH"):
                return await self._handle_update(
                    scope, receive, send, cql2_filter, path, method
                )
            if method == "DELETE":
                return await self._handle_delete(
                    scope, receive, send, cql2_filter, path
                )

        # Match catalog children endpoints: /catalogs/{id}/catalogs,
        # /catalogs/{id}/collections, /catalogs/{id}/{catalogs|collections}/{id}
        match = re.match(self.catalog_children_pattern, path)
        if match:
            catalog_id, children, child_id = match.groups()
            catalog_path = f"/catalogs/{quote(catalog_id, safe='')}"
            if method == "POST" and child_id is None:
                return await self._handle_add_child(
                    scope, receive, send, cql2_filter, catalog_path, children
                )
            if (
                method in ("PUT", "PATCH")
                and children == "collections"
                and child_id is not None
            ):
                return await self._handle_update(
                    scope, receive, send, cql2_filter, path, method
                )
            if method == "DELETE" and child_id is not None:
                return await self._handle_remove_child(
                    scope,
                    receive,
                    send,
                    cql2_filter,
                    catalog_path,
                    f"/{children}/{quote(child_id, safe='')}",
                )

        # Not a transaction endpoint, pass through
        return await self.app(scope, receive, send)

    async def _read_body(self, receive: Receive) -> bytes:
        """Read the full request body."""
        body = b""
        more_body = True
        while more_body:
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
                more_body = message.get("more_body", False)
        return body

    def _make_receive(self, body: bytes) -> Receive:
        """Create a new receive callable that returns the given body."""

        async def new_receive():
            return {
                "type": "http.request",
                "body": body,
                "more_body": False,
            }

        return new_receive

    async def _fetch_existing(self, path: str) -> Optional[dict]:
        """Fetch the existing record from upstream."""
        response = await self._client.get(path)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def _denied_existing(self, scope: Scope, existing: dict) -> JSONResponse:
        """
        Refuse a change to an existing record that the filter does not match.

        A record the caller may read is refused with 403; any other record is
        reported as not found, so its existence is not disclosed.
        """
        read_filter: Optional[Expr] = getattr(
            Request(scope).state, self.read_state_key, None
        )
        if read_filter is not None and self._readable(read_filter, existing):
            return JSONResponse(
                {
                    "code": "ForbiddenError",
                    "description": "Resource does not match access filter.",
                },
                status_code=403,
            )
        return JSONResponse(
            {"code": "NotFoundError", "description": "Record not found."},
            status_code=404,
        )

    @staticmethod
    def _readable(read_filter: Expr, record: dict) -> bool:
        """Check the read filter against a record; an evaluation error counts as no match."""
        try:
            return bool(read_filter.matches(record))
        except Exception as e:
            logger.warning("Could not evaluate the read filter on a record: %s", e)
            return False

    async def _handle_create(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        cql2_filter: Expr,
    ) -> None:
        """Validate create requests."""
        body = await self._read_body(receive)

        try:
            body_json = json.loads(body) if body else {}
        except json.JSONDecodeError:
            response = JSONResponse(
                {
                    "code": "ParseError",
                    "description": "Request body must be valid JSON.",
                },
                status_code=400,
            )
            return await response(scope, receive, send)

        if not cql2_filter.matches(body_json):
            response = JSONResponse(
                {
                    "code": "ForbiddenError",
                    "description": "Resource does not match access filter.",
                },
                status_code=403,
            )
            return await response(scope, receive, send)

        # Reconstruct receive and forward
        scope = dict(scope)
        await self.app(scope, self._make_receive(body), send)

    async def _handle_bulk_create(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        cql2_filter: Expr,
    ) -> None:
        """Validate bulk item create requests."""
        body = await self._read_body(receive)

        try:
            body_json = json.loads(body) if body else {}
        except json.JSONDecodeError:
            response = JSONResponse(
                {
                    "code": "ParseError",
                    "description": "Request body must be valid JSON.",
                },
                status_code=400,
            )
            return await response(scope, receive, send)

        items = body_json.get("items", {})
        if not isinstance(items, dict):
            response = JSONResponse(
                {
                    "code": "ParseError",
                    "description": "Bulk items body must contain an 'items' object.",
                },
                status_code=400,
            )
            return await response(scope, receive, send)

        failed = [
            item_id for item_id, item in items.items() if not cql2_filter.matches(item)
        ]

        if failed:
            response = JSONResponse(
                {
                    "code": "ForbiddenError",
                    "description": f"Items do not match access filter: {', '.join(failed)}",
                },
                status_code=403,
            )
            return await response(scope, receive, send)

        # Reconstruct receive and forward
        scope = dict(scope)
        await self.app(scope, self._make_receive(body), send)

    async def _handle_update(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        cql2_filter: Expr,
        path: str,
        method: str,
    ) -> None:
        """Validate update requests."""
        body = await self._read_body(receive)

        try:
            body_json = json.loads(body) if body else {}
        except json.JSONDecodeError:
            response = JSONResponse(
                {
                    "code": "ParseError",
                    "description": "Request body must be valid JSON.",
                },
                status_code=400,
            )
            return await response(scope, receive, send)

        # Fetch existing record
        try:
            existing = await self._fetch_existing(path)
        except httpx.HTTPError:
            response = JSONResponse(
                {
                    "code": "UpstreamError",
                    "description": "Failed to fetch record from upstream.",
                },
                status_code=502,
            )
            return await response(scope, receive, send)

        if existing is None:
            response = JSONResponse(
                {"code": "NotFoundError", "description": "Record not found."},
                status_code=404,
            )
            return await response(scope, receive, send)

        # Validate existing record matches filter
        if not cql2_filter.matches(existing):
            response = self._denied_existing(scope, existing)
            return await response(scope, receive, send)

        # Merge for validation
        if method == "PATCH":
            merged = _deep_merge(existing, body_json)
        else:
            merged = body_json

        # Validate merged result matches filter
        if not cql2_filter.matches(merged):
            response = JSONResponse(
                {
                    "code": "ForbiddenError",
                    "description": "Updated resource does not match access filter.",
                },
                status_code=403,
            )
            return await response(scope, receive, send)

        # Forward
        scope = dict(scope)
        await self.app(scope, self._make_receive(body), send)

    async def _handle_delete(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        cql2_filter: Expr,
        path: str,
    ) -> None:
        """Validate delete requests."""
        try:
            existing = await self._fetch_existing(path)
        except httpx.HTTPError:
            response = JSONResponse(
                {
                    "code": "UpstreamError",
                    "description": "Failed to fetch record from upstream.",
                },
                status_code=502,
            )
            return await response(scope, receive, send)

        if existing is None:
            response = JSONResponse(
                {"code": "NotFoundError", "description": "Record not found."},
                status_code=404,
            )
            return await response(scope, receive, send)

        if not cql2_filter.matches(existing):
            response = self._denied_existing(scope, existing)
            return await response(scope, receive, send)

        await self.app(scope, receive, send)

    async def _check_existing(
        self, cql2_filter: Expr, path: str
    ) -> Optional[JSONResponse]:
        """Check that an existing record matches the filter; return the refusal if not."""
        try:
            existing = await self._fetch_existing(path)
        except httpx.HTTPError:
            return JSONResponse(
                {
                    "code": "UpstreamError",
                    "description": "Failed to fetch record from upstream.",
                },
                status_code=502,
            )

        if existing is None or not cql2_filter.matches(existing):
            return JSONResponse(
                {"code": "NotFoundError", "description": "Record not found."},
                status_code=404,
            )
        return None

    async def _handle_add_child(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        cql2_filter: Expr,
        catalog_path: str,
        children: str,
    ) -> None:
        """
        Validate adding a catalog or collection to a catalog.

        The body either creates a new record or links an existing one by its id.
        The catalog must match the filter. Linking changes the linked record, so an
        existing record with the body's id must match the filter; otherwise the body
        must match, as for any create.
        """
        body = await self._read_body(receive)

        try:
            body_json = json.loads(body) if body else {}
        except json.JSONDecodeError:
            response = JSONResponse(
                {
                    "code": "ParseError",
                    "description": "Request body must be valid JSON.",
                },
                status_code=400,
            )
            return await response(scope, receive, send)

        child_id = body_json.get("id") if isinstance(body_json, dict) else None
        if not isinstance(child_id, str) or not child_id or "/" in child_id:
            response = JSONResponse(
                {
                    "code": "ParseError",
                    "description": "Request body must have a string id without '/'.",
                },
                status_code=400,
            )
            return await response(scope, receive, send)

        denied = await self._check_existing(cql2_filter, catalog_path)
        if denied:
            return await denied(scope, receive, send)

        try:
            existing = await self._fetch_existing(
                f"/{children}/{quote(child_id, safe='')}"
            )
        except httpx.HTTPError:
            response = JSONResponse(
                {
                    "code": "UpstreamError",
                    "description": "Failed to fetch record from upstream.",
                },
                status_code=502,
            )
            return await response(scope, receive, send)

        # A linked record the caller cannot change gets the same answer as a new
        # record the caller cannot create, so the answer does not tell them apart.
        if not cql2_filter.matches(existing if existing is not None else body_json):
            response = JSONResponse(
                {
                    "code": "ForbiddenError",
                    "description": "Resource does not match access filter.",
                },
                status_code=403,
            )
            return await response(scope, receive, send)

        # Reconstruct receive and forward
        scope = dict(scope)
        await self.app(scope, self._make_receive(body), send)

    async def _handle_remove_child(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        cql2_filter: Expr,
        catalog_path: str,
        child_path: str,
    ) -> None:
        """
        Validate unlinking a child, which changes both the catalog and the child.

        The child, which the path names, is checked first, so a refusal answers for it.
        """
        denied = await self._check_existing(
            cql2_filter, child_path
        ) or await self._check_existing(cql2_filter, catalog_path)
        if denied:
            return await denied(scope, receive, send)

        await self.app(scope, receive, send)
