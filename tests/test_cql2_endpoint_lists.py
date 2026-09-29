"""Test the configurable endpoint lists of the CQL2 filter middlewares."""

import json

import pytest
from cql2 import Expr
from fastapi import FastAPI, Request
from starlette.testclient import TestClient

from stac_auth_proxy.config import (
    DEFAULT_SEARCH_BODY_ENDPOINTS,
    DEFAULT_SINGLE_RECORD_ENDPOINTS,
    Settings,
)
from stac_auth_proxy.middleware.Cql2ApplyFilterBodyMiddleware import (
    Cql2ApplyFilterBodyMiddleware,
)
from stac_auth_proxy.middleware.Cql2ApplyFilterQueryStringMiddleware import (
    Cql2ApplyFilterQueryStringMiddleware,
)
from stac_auth_proxy.middleware.Cql2ValidateResponseBodyMiddleware import (
    Cql2ValidateResponseBodyMiddleware,
)

TENANT_A = Expr("tenant = 'a'")
TENANT_B = Expr("tenant = 'b'")
CATALOG = r"^/catalogs/([^/]+)$"
CATALOG_SEARCH = r"^/catalogs/([^/]+)/search$"


def _app(middleware, cql2_filter: Expr, **kwargs) -> TestClient:
    """Build an app that echoes the request, behind one middleware and a fixed filter."""
    app = FastAPI()
    app.add_middleware(middleware, **kwargs)

    @app.middleware("http")
    async def set_filter(request, call_next):
        request.state.cql2_filter = cql2_filter
        return await call_next(request)

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def echo(request: Request):
        body = await request.body()
        return {
            "id": "c1",
            "tenant": "a",
            "query": dict(request.query_params),
            "body": json.loads(body) if body else None,
        }

    return TestClient(app)


class TestQueryStringFilter:
    """Cql2ApplyFilterQueryStringMiddleware leaves single-record endpoints alone."""

    @pytest.mark.parametrize(
        "kwargs,path,filtered",
        [
            pytest.param({}, "/collections", True, id="default-list"),
            pytest.param({}, "/collections/c1", False, id="default-single"),
            pytest.param({}, "/catalogs/c1", True, id="default-other"),
            pytest.param(
                {"single_record_endpoints": [CATALOG]},
                "/catalogs/c1",
                False,
                id="configured-single",
            ),
            pytest.param(
                {"single_record_endpoints": [CATALOG]},
                "/catalogs",
                True,
                id="configured-list",
            ),
        ],
    )
    def test_filter_added_only_to_lists(self, kwargs, path, filtered):
        """The filter goes into the query string unless the path is a single record."""
        client = _app(Cql2ApplyFilterQueryStringMiddleware, TENANT_A, **kwargs)
        query = client.get(path).json()["query"]
        assert ("filter" in query) is filtered


class TestResponseBodyValidation:
    """Cql2ValidateResponseBodyMiddleware checks the configured single-record endpoints."""

    @pytest.mark.parametrize(
        "kwargs,cql2_filter,expected_status",
        [
            pytest.param({}, TENANT_B, 200, id="default-not-checked"),
            pytest.param(
                {"single_record_endpoints": [CATALOG]},
                TENANT_A,
                200,
                id="configured-match",
            ),
            pytest.param(
                {"single_record_endpoints": [CATALOG]},
                TENANT_B,
                404,
                id="configured-no-match",
            ),
        ],
    )
    def test_configured_single_records_are_checked(
        self, kwargs, cql2_filter, expected_status
    ):
        """A record that does not match the filter is hidden on configured paths."""
        client = _app(Cql2ValidateResponseBodyMiddleware, cql2_filter, **kwargs)
        assert client.get("/catalogs/c1").status_code == expected_status


class TestBodyFilter:
    """Cql2ApplyFilterBodyMiddleware adds the filter to configured search bodies."""

    @pytest.mark.parametrize(
        "kwargs,path,filtered",
        [
            pytest.param({}, "/search", True, id="default-search"),
            pytest.param({}, "/catalogs/c1/search", False, id="default-other"),
            pytest.param(
                {"search_body_endpoints": [CATALOG_SEARCH]},
                "/catalogs/c1/search",
                True,
                id="configured-search",
            ),
        ],
    )
    def test_filter_added_to_search_bodies(self, kwargs, path, filtered):
        """The filter goes into the POST body of the configured search endpoints."""
        client = _app(Cql2ApplyFilterBodyMiddleware, TENANT_A, **kwargs)
        body = client.post(path, json={"limit": 1}).json()["body"]
        assert ("filter" in body) is filtered


class TestSettings:
    """The endpoint lists are settings with the previous lists as defaults."""

    def test_defaults(self):
        """Without settings, the lists are the ones the middlewares always used."""
        settings = Settings(
            upstream_url="https://example.com",
            oidc_discovery_url="https://example.com/.well-known/openid-configuration",
        )
        assert list(settings.single_record_endpoints) == DEFAULT_SINGLE_RECORD_ENDPOINTS
        assert list(settings.search_body_endpoints) == DEFAULT_SEARCH_BODY_ENDPOINTS

    def test_from_environment(self, monkeypatch):
        """The lists are read from JSON environment variables."""
        monkeypatch.setenv("SINGLE_RECORD_ENDPOINTS", json.dumps([CATALOG]))
        monkeypatch.setenv("SEARCH_BODY_ENDPOINTS", json.dumps([CATALOG_SEARCH]))
        settings = Settings(
            upstream_url="https://example.com",
            oidc_discovery_url="https://example.com/.well-known/openid-configuration",
        )
        assert list(settings.single_record_endpoints) == [CATALOG]
        assert list(settings.search_body_endpoints) == [CATALOG_SEARCH]

    def test_app_passes_the_lists_to_the_middlewares(self, source_api_server):
        """The configured lists reach the middlewares that use them."""
        from utils import AppFactory

        app = AppFactory(
            oidc_discovery_url="https://example-stac-api.com/.well-known/openid-configuration",
            items_filter={"cls": "stac_auth_proxy.filters:Template", "args": ["true"]},
            single_record_endpoints=[CATALOG],
            search_body_endpoints=[CATALOG_SEARCH],
        )(upstream_url=source_api_server)
        kwargs = {m.cls: m.kwargs for m in app.user_middleware}
        assert kwargs[Cql2ValidateResponseBodyMiddleware][
            "single_record_endpoints"
        ] == [CATALOG]
        assert kwargs[Cql2ApplyFilterQueryStringMiddleware][
            "single_record_endpoints"
        ] == [CATALOG]
        assert kwargs[Cql2ApplyFilterBodyMiddleware]["search_body_endpoints"] == [
            CATALOG_SEARCH
        ]
