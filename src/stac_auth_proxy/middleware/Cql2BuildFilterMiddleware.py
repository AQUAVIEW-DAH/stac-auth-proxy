"""Middleware to build the Cql2Filter."""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Sequence

from cql2 import Expr, ValidationError
from fastapi import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from ..utils import requests
from ..utils.middleware import required_conformance

logger = logging.getLogger(__name__)


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
    parents_state_key: str = "cql2_parent_filters"

    # Filters
    collections_filter: Optional[Callable] = None
    collections_filter_path: str = r"^/collections(/[^/]+)?$"
    items_filter: Optional[Callable] = None
    items_filter_path: str = r"^(/collections/([^/]+)/items(/[^/]+)?$|/search$)"

    # Sub-resource path patterns, each mapped to the paths of its parent records
    sub_resource_endpoints: dict[str, Sequence[str]] = field(default_factory=dict)

    def __post_init__(self):
        """Set required conformances based on the filter functions."""
        for pattern, parent_paths in self.sub_resource_endpoints.items():
            for parent_path in parent_paths:
                example = parent_path.format_map(
                    {name: "x" for name in re.compile(pattern).groupindex}
                )
                if not self._get_filter(example):
                    logger.warning(
                        "No filter covers parent path %r of %r; it is not checked.",
                        parent_path,
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

        if request.method.upper() == "OPTIONS":
            logger.debug("Skipping CQL2 filter build for OPTIONS request")
            return await self.app(scope, receive, send)

        parent_paths = self._parent_paths(request.url.path)
        if parent_paths:
            try:
                parent_filters = await self._build_parent_filters(
                    request, scope, parent_paths
                )
            except HTTPException as e:
                response = JSONResponse(
                    {"detail": e.detail}, status_code=e.status_code, headers=e.headers
                )
                return await response(scope, receive, send)
            except ValidationError:
                logger.error("Invalid CQL2 filter for a parent of %s", request.url.path)
                response = JSONResponse(
                    {"detail": "Invalid CQL2 filter"}, status_code=502
                )
                return await response(scope, receive, send)
            setattr(request.state, self.parents_state_key, parent_filters)

        filter_builder = self._get_filter(request.url.path)
        if not filter_builder:
            return await self.app(scope, receive, send)

        try:
            filter_expr = await filter_builder(
                {
                    "req": {
                        "path": request.url.path,
                        "method": request.method,
                        "query_params": dict(request.query_params),
                        "path_params": requests.extract_variables(request.url.path),
                        "headers": dict(request.headers),
                    },
                    **scope["state"],
                }
            )
        except HTTPException as e:
            response = JSONResponse({"detail": e.detail}, status_code=e.status_code)
            return await response(scope, receive, send)

        cql2_filter = Expr(filter_expr)
        try:
            cql2_filter.validate()
        except ValidationError:
            logger.error("Invalid CQL2 filter: %s", filter_expr)
            response = JSONResponse({"detail": "Invalid CQL2 filter"}, status_code=502)
            return await response(scope, receive, send)

        setattr(request.state, self.state_key, cql2_filter)

        return await self.app(scope, receive, send)

    def _parent_paths(self, path: str) -> list[str]:
        """Get the parent record paths of a sub-resource path."""
        for pattern, parent_paths in self.sub_resource_endpoints.items():
            match = re.match(pattern, path)
            if match:
                groups = match.groupdict()
                return [parent_path.format_map(groups) for parent_path in parent_paths]
        return []

    async def _build_parent_filters(
        self, request: Request, scope: Scope, parent_paths: list[str]
    ) -> list[tuple[str, Expr]]:
        """Build the filter the caller would get for reading each parent record."""
        parent_filters = []
        for parent_path in parent_paths:
            filter_builder = self._get_filter(parent_path)
            if not filter_builder:
                continue
            parent_filter = Expr(
                await filter_builder(
                    {
                        "req": {
                            "path": parent_path,
                            "method": "GET",
                            "query_params": {},
                            "path_params": requests.extract_variables(parent_path),
                            "headers": dict(request.headers),
                        },
                        **scope["state"],
                    }
                )
            )
            parent_filter.validate()
            parent_filters.append((parent_path, parent_filter))
        return parent_filters

    def _get_filter(
        self, path: str
    ) -> Optional[Callable[..., Awaitable[str | dict[str, Any]]]]:
        """Get the CQL2 filter builder for the given path."""
        endpoint_filters = [
            (self.collections_filter_path, self.collections_filter),
            (self.items_filter_path, self.items_filter),
        ]
        for expr, builder in endpoint_filters:
            if re.match(expr, path):
                return builder
        return None
