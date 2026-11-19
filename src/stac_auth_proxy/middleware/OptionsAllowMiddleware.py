"""Middleware to answer OPTIONS requests with the methods the caller may use."""

import re
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Optional, Sequence

import httpx
from cql2 import Expr
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from ..config import EndpointMethods
from ..utils.requests import find_match, is_cors_preflight

logger = getLogger(__name__)

READ_METHODS = ("GET", "HEAD")
WRITE_METHODS = ("POST", "PUT", "PATCH", "DELETE")


@dataclass
class OptionsAllowMiddleware:
    """
    Answer non-preflight OPTIONS requests with an Allow header.

    The header lists the methods the caller may use on the resource, following
    https://www.rfc-editor.org/rfc/rfc9110#section-9.3.7 and OGC API - Features -
    Part 4. A method is listed if it passes the endpoint's authentication and scope
    checks and, where a CQL2 filter applies, the caller's filter for that method:

    - On a single record, the filter must match the record, fetched from upstream.
      A record the caller may not read is reported as not found, as a GET would be.
    - On other paths, the filter must have been built. The body of a create is not
      known, so it is checked on the request itself.

    Write methods are listed only where PRIVATE_ENDPOINTS names them for the path.
    CORS preflight requests are passed on unchanged.
    """

    app: ASGIApp
    upstream_url: str
    private_endpoints: EndpointMethods
    public_endpoints: EndpointMethods
    default_public: bool
    single_record_endpoints: Sequence[str] = field(
        default_factory=lambda: [
            r"^/collections/([^/]+)/items/([^/]+)$",
            r"^/collections/([^/]+)$",
        ]
    )
    state_key: str = "payload"
    filters_state_key: str = "cql2_filters"

    _client: httpx.AsyncClient = field(init=False)

    def __post_init__(self):
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(base_url=self.upstream_url)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Answer an OPTIONS request with the methods the caller may use."""
        if scope["type"] != "http" or scope["method"] != "OPTIONS":
            return await self.app(scope, receive, send)

        request = Request(scope)
        if is_cors_preflight(request):
            return await self.app(scope, receive, send)

        path = request.url.path
        payload = getattr(request.state, self.state_key, None)
        methods = [
            method
            for method in (*READ_METHODS, *self._write_methods(path))
            if self._authorized(path, method, payload)
        ]

        filters: Optional[dict[str, Expr]] = getattr(
            request.state, self.filters_state_key, None
        )
        if filters is not None and methods:
            if any(re.match(expr, path) for expr in self.single_record_endpoints):
                try:
                    record = await self._fetch_record(path)
                except httpx.HTTPError:
                    response: Response = JSONResponse(
                        {
                            "code": "UpstreamError",
                            "description": "Failed to fetch record from upstream.",
                        },
                        status_code=502,
                    )
                    return await response(scope, receive, send)

                if "GET" in methods and not (
                    record is not None and self._matches(filters.get("GET"), record)
                ):
                    response = JSONResponse(
                        {"code": "NotFoundError", "description": "Record not found."},
                        status_code=404,
                    )
                    return await response(scope, receive, send)

                methods = [
                    method
                    for method in methods
                    if record is not None
                    and self._matches(filters.get(_filter_method(method)), record)
                ]
            else:
                methods = [
                    method for method in methods if _filter_method(method) in filters
                ]

        response = Response(headers={"Allow": ", ".join([*methods, "OPTIONS"])})
        return await response(scope, receive, send)

    def _write_methods(self, path: str) -> list[str]:
        """Get the write methods that PRIVATE_ENDPOINTS names for the path."""
        named = set()
        for pattern, endpoint_methods in self.private_endpoints.items():
            if re.match(pattern, path):
                for endpoint_method in endpoint_methods:
                    if isinstance(endpoint_method, tuple):
                        endpoint_method = endpoint_method[0]
                    named.add(endpoint_method.upper())
        return [method for method in WRITE_METHODS if method in named]

    def _authorized(
        self, path: str, method: str, payload: Optional[dict[str, Any]]
    ) -> bool:
        """Check the endpoint's authentication and scope requirements for a method."""
        match = find_match(
            path,
            method,
            private_endpoints=self.private_endpoints,
            public_endpoints=self.public_endpoints,
            default_public=self.default_public,
        )
        if not match.uses_auth:
            return True
        if payload is None:
            return False
        token_scopes = set(payload.get("scope", "").split())
        return set(match.required_scopes) <= token_scopes

    async def _fetch_record(self, path: str) -> Optional[dict]:
        """Fetch the record from upstream."""
        response = await self._client.get(path)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _matches(cql2_filter: Optional[Expr], record: dict) -> bool:
        """Check a filter against a record; a missing filter or an evaluation error counts as no match."""
        if cql2_filter is None:
            return False
        try:
            return bool(cql2_filter.matches(record))
        except Exception as e:
            logger.warning("Could not evaluate a filter on a record: %s", e)
            return False


def _filter_method(method: str) -> str:
    """Get the method whose filter decides a method; HEAD reads like GET."""
    return "GET" if method == "HEAD" else method
