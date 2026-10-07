"""Middleware to build the Cql2Filter."""

import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Sequence

from cql2 import Expr, ValidationError
from fastapi import HTTPException
from starlette.datastructures import QueryParams
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ..config import DEFAULT_COLLECTIONS_FILTER_PATH, DEFAULT_ITEMS_FILTER_PATH
from ..utils import filters, requests
from ..utils.middleware import bad_request, required_conformance
from ..utils.requests import match_path

logger = logging.getLogger(__name__)

# Methods whose filters are built for a non-preflight OPTIONS request
OPTIONS_FILTER_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")


@required_conformance(
    "http://www.opengis.net/spec/cql2/1.0/conf/basic-cql2",
    "http://www.opengis.net/spec/cql2/1.0/conf/cql2-text",
    "http://www.opengis.net/spec/cql2/1.0/conf/cql2-json",
)
@dataclass(frozen=True)
class Cql2BuildFilterMiddleware:
    """Middleware to build the Cql2Filter."""

    app: ASGIApp

    state_key: str = "cql2_filter"
    read_state_key: str = "cql2_read_filter"

    # On a non-preflight OPTIONS request, build the filter for each method and place
    # them in request state (used by OptionsAllowMiddleware)
    options_filters: bool = False
    options_state_key: str = "cql2_filters"

    # Filters
    collections_filter: Optional[Callable] = None
    collections_filter_path: str | Sequence[str] = (DEFAULT_COLLECTIONS_FILTER_PATH,)
    items_filter: Optional[Callable] = None
    items_filter_path: str | Sequence[str] = (DEFAULT_ITEMS_FILTER_PATH,)

    def __post_init__(self):
        """Set required conformances based on the filter functions."""
        for attr in ("collections_filter_path", "items_filter_path"):
            object.__setattr__(self, attr, requests.as_patterns(getattr(self, attr)))
            for pattern in getattr(self, attr):
                if not re.compile(pattern).groupindex:
                    logger.info(
                        "Filter path %r declares no named capture groups, "
                        "falling back to built-in path param extraction.",
                        pattern,
                    )

        required_conformances = set()
        if self.collections_filter:
            logger.debug("Appending required conformance for collections filter")
            # https://github.com/stac-api-extensions/collection-search/blob/4825b4b1cee96bdc0cbfbb342d5060d0031976f0/README.md#L5
            required_conformances.update(
                [
                    "https://api.stacspec.org/v1.0.0/core",
                    r"https://api.stacspec.org/v1\.0\.0(?:-[\w\.]+)?/collection-search",
                    r"https://api.stacspec.org/v1\.0\.0(?:-[\w\.]+)?/collection-search#filter",
                    "http://www.opengis.net/spec/ogcapi-common-2/1.0/conf/simple-query",
                ]
            )
        if self.items_filter:
            logger.debug("Appending required conformance for items filter")
            # https://github.com/stac-api-extensions/filter/blob/c763dbbf0a52210ab8d9866ff048da448d270f93/README.md#conformance-classes
            required_conformances.update(
                [
                    "http://www.opengis.net/spec/ogcapi-features-3/1.0/conf/filter",
                    "http://www.opengis.net/spec/ogcapi-features-3/1.0/conf/features-filter",
                    r"https://api.stacspec.org/v1\.0\.0(?:-[\w\.]+)?/item-search#filter",
                ]
            )

        # Must set required conformances on class
        self.__class__.__required_conformances__ = required_conformances.union(
            getattr(self.__class__, "__required_conformances__", [])
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Build the CQL2 filter, place on the request state."""
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request = Request(scope)

        is_options = request.method.upper() == "OPTIONS"
        if is_options and (
            not self.options_filters or requests.is_cors_preflight(request)
        ):
            logger.debug("Skipping CQL2 filter build for OPTIONS request")
            return await self.app(scope, receive, send)

        filter_builder, path_params = self._get_filter(request.url.path)
        if not filter_builder:
            return await self.app(scope, receive, send)

        # The filter builder sees one value per key, while the upstream may act on a
        # different one. Fail closed rather than let them disagree.
        query_string = scope.get("query_string", b"")
        query_params = filters.parse_query_params(query_string)
        try:
            filters.check_query_size(query_string)
            filters.check_unique_params(k for k, _ in query_params.multi_items())
        except filters.InvalidFilterRequestError as e:
            return await bad_request(str(e))(scope, receive, send)

        if is_options:
            options_filters = await self._build_options_filters(
                filter_builder, request, scope, query_params, path_params
            )
            setattr(request.state, self.options_state_key, options_filters)
            return await self.app(scope, receive, send)

        try:
            filter_expr = await filter_builder(
                self._context(request, scope, request.method, query_params, path_params)
            )
        except HTTPException as e:
            response = JSONResponse(
                {"detail": e.detail}, status_code=e.status_code, headers=e.headers
            )
            return await response(scope, receive, send)

        cql2_filter = Expr(filter_expr)
        try:
            cql2_filter.validate()
        except ValidationError:
            logger.error("Invalid CQL2 filter: %s", filter_expr)
            response = JSONResponse({"detail": "Invalid CQL2 filter"}, status_code=502)
            return await response(scope, receive, send)

        setattr(request.state, self.state_key, cql2_filter)

        if request.method.upper() in ("PUT", "PATCH", "DELETE"):
            # Lets transaction validation tell a record the caller may read but not
            # modify (403) from one the caller may not see (404).
            read_filter = await self._build_read_filter(
                filter_builder, request, scope, query_params, path_params
            )
            if read_filter is not None:
                setattr(request.state, self.read_state_key, read_filter)

        return await self.app(scope, receive, send)

    @staticmethod
    def _context(
        request: Request,
        scope: Scope,
        method: str,
        query_params: QueryParams,
        path_params: dict,
    ) -> dict[str, Any]:
        """Build the context passed to a filter builder."""
        return {
            "req": {
                "path": request.url.path,
                "method": method,
                "query_params": dict(query_params),
                "path_params": path_params,
                "headers": dict(request.headers),
            },
            **scope["state"],
        }

    async def _build_read_filter(
        self,
        filter_builder: Callable[..., Awaitable[str | dict[str, Any]]],
        request: Request,
        scope: Scope,
        query_params: QueryParams,
        path_params: dict,
    ) -> Optional[Expr]:
        """Build the filter the caller would get for reading the same path."""
        try:
            read_filter = Expr(
                await filter_builder(
                    self._context(request, scope, "GET", query_params, path_params)
                )
            )
            read_filter.validate()
        except (HTTPException, ValidationError) as e:
            logger.debug("No read filter for %s: %s", request.url.path, e)
            return None
        return read_filter

    async def _build_options_filters(
        self,
        filter_builder: Callable[..., Awaitable[str | dict[str, Any]]],
        request: Request,
        scope: Scope,
        query_params: QueryParams,
        path_params: dict,
    ) -> dict[str, Expr]:
        """Build the filter the caller would get for each method on the same path."""
        options_filters = {}
        for method in OPTIONS_FILTER_METHODS:
            try:
                cql2_filter = Expr(
                    await filter_builder(
                        self._context(request, scope, method, query_params, path_params)
                    )
                )
                cql2_filter.validate()
            except (HTTPException, ValidationError) as e:
                logger.debug("No %s filter for %s: %s", method, request.url.path, e)
                continue
            options_filters[method] = cql2_filter
        return options_filters

    def _get_filter(
        self, path: str
    ) -> tuple[Optional[Callable[..., Awaitable[str | dict[str, Any]]]], dict]:
        """Get the CQL2 filter builder for the given path and its path params."""
        endpoint_filters = [
            (self.collections_filter_path, self.collections_filter),
            (self.items_filter_path, self.items_filter),
        ]
        for patterns, builder in endpoint_filters:
            for expr in patterns:
                match = match_path(expr, path)
                if match:
                    return builder, self._path_params(match, path)
        return None, {}

    @staticmethod
    def _path_params(match: re.Match, path: str) -> dict:
        """Get the path params declared by a matched pattern's named groups."""
        if match.re.groupindex:
            return {k: v for k, v in match.groupdict().items() if v is not None}
        return requests.extract_variables(path)
