"""Middleware to check the parent records of a sub-resource against CQL2 filters."""

from dataclasses import dataclass, field
from logging import getLogger
from typing import Optional

import httpx
from cql2 import Expr
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ..utils.middleware import required_conformance

logger = getLogger(__name__)


@required_conformance(
    r"http://www.opengis.net/spec/cql2/1.0/conf/basic-cql2",
    r"http://www.opengis.net/spec/cql2/1.0/conf/cql2-text",
    r"http://www.opengis.net/spec/cql2/1.0/conf/cql2-json",
)
@dataclass
class Cql2ValidateParentRecordsMiddleware:
    """
    Serve a sub-resource only when the caller may read each of its parent records.

    Cql2BuildFilterMiddleware places the parent record paths and their filters on the
    request state. Each parent is fetched from the upstream API and checked against
    its filter. A parent that is missing, does not match, or cannot be evaluated is
    answered with the same 404, so the response does not tell a hidden parent from a
    missing one.
    """

    app: ASGIApp
    upstream_url: str
    state_key: str = "cql2_parent_filters"

    _client: httpx.AsyncClient = field(init=False)

    def __post_init__(self):
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(base_url=self.upstream_url)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Check each parent record before the sub-resource is served."""
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        parent_filters: list[tuple[str, Expr]] = (
            getattr(Request(scope).state, self.state_key, None) or []
        )

        for parent_path, cql2_filter in parent_filters:
            try:
                parent = await self._fetch_parent(parent_path)
            except (httpx.HTTPError, ValueError):
                response = JSONResponse(
                    {
                        "code": "UpstreamError",
                        "description": "Failed to fetch record from upstream.",
                    },
                    status_code=502,
                )
                return await response(scope, receive, send)

            if parent is None or not self._matches(cql2_filter, parent):
                response = JSONResponse(
                    {"code": "NotFoundError", "description": "Record not found."},
                    status_code=404,
                )
                return await response(scope, receive, send)

        return await self.app(scope, receive, send)

    async def _fetch_parent(self, path: str) -> Optional[dict]:
        """Fetch a parent record from upstream."""
        response = await self._client.get(httpx.URL(path=path))
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _matches(cql2_filter: Expr, record: dict) -> bool:
        """Check a filter against a record; an evaluation error counts as no match."""
        try:
            return bool(cql2_filter.matches(record))
        except Exception as e:
            logger.warning("Could not evaluate the filter on a parent record: %s", e)
            return False
