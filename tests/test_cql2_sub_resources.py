"""Test the parent record checks for sub-resource endpoints."""

import os
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from cql2 import Expr
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError
from utils import AppFactory

from stac_auth_proxy.config import Settings
from stac_auth_proxy.middleware.Cql2BuildFilterMiddleware import (
    Cql2BuildFilterMiddleware,
)
from stac_auth_proxy.middleware.Cql2ValidateParentRecordsMiddleware import (
    Cql2ValidateParentRecordsMiddleware,
)

QUERYABLES = r"^/collections/(?P<collection_id>[^/]+)/queryables$"
SUB_RESOURCES = {QUERYABLES: ["/collections/{collection_id}"]}
COMMON = {
    "upstream_url": "https://example.com",
    "oidc_discovery_url": "https://example.com/.well-known/openid-configuration",
}


class TestSettings:
    """SUB_RESOURCE_ENDPOINTS is parsed and checked at startup."""

    def test_default_is_empty(self):
        """No sub-resource endpoints are configured by default."""
        assert Settings(**COMMON).sub_resource_endpoints == {}

    def test_parsed_from_the_environment(self):
        """The setting is read as a JSON object."""
        env = {
            "SUB_RESOURCE_ENDPOINTS": '{"^/collections/(?P<collection_id>[^/]+)/queryables$": ["/collections/{collection_id}"]}'
        }
        with patch.dict(os.environ, env):
            assert Settings(**COMMON).sub_resource_endpoints == SUB_RESOURCES

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param({"^/collections/([^/]+$": ["/x"]}, id="invalid-regex"),
            pytest.param({QUERYABLES: ["/catalogs/{catalog_id}"]}, id="unknown-group"),
            pytest.param(
                {QUERYABLES: ["collections/{collection_id}"]}, id="relative-path"
            ),
        ],
    )
    def test_invalid_values_are_rejected(self, value):
        """A pattern that does not compile or a parent it cannot fill fails early."""
        with pytest.raises(ValidationError):
            Settings(**COMMON, sub_resource_endpoints=value)


def _build_app(collections_filter, **kwargs) -> TestClient:
    """Build an app that returns the parent filters placed on the request state."""
    app = FastAPI()
    app.add_middleware(
        Cql2BuildFilterMiddleware,
        collections_filter=collections_filter,
        **kwargs,
    )

    @app.api_route("/{path:path}", methods=["GET", "POST", "OPTIONS"])
    async def echo(request: Request):
        parents = getattr(request.state, "cql2_parent_filters", None)
        if parents is None:
            return {"parents": None}
        return {"parents": [[path, expr.to_text()] for path, expr in parents]}

    return TestClient(app)


class TestBuildParentFilters:
    """Cql2BuildFilterMiddleware builds the filter of each parent record."""

    def test_parent_filter_is_built_for_a_read_of_the_parent(self):
        """The filter factory is asked about a GET of the parent path."""
        contexts = []

        async def collections_filter(context):
            contexts.append(context["req"])
            return "private = false"

        client = _build_app(collections_filter, sub_resource_endpoints=SUB_RESOURCES)
        response = client.post("/collections/c1/queryables", params={"q": "x"})
        assert response.json()["parents"] == [
            ["/collections/c1", Expr("private = false").to_text()]
        ]
        assert [(c["path"], c["method"], c["query_params"]) for c in contexts] == [
            ("/collections/c1", "GET", {})
        ]
        assert contexts[0]["path_params"] == {"collection_id": "c1"}

    @pytest.mark.parametrize(
        "kwargs,method,path",
        [
            pytest.param({}, "GET", "/collections/c1/queryables", id="not-configured"),
            pytest.param(
                {"sub_resource_endpoints": SUB_RESOURCES},
                "GET",
                "/collections/c1/other",
                id="other-path",
            ),
            pytest.param(
                {"sub_resource_endpoints": SUB_RESOURCES},
                "OPTIONS",
                "/collections/c1/queryables",
                id="options",
            ),
        ],
    )
    def test_no_parent_filters(self, kwargs, method, path):
        """Paths that are not configured sub-resources get no parent filters."""

        async def collections_filter(context):
            return "private = false"

        client = _build_app(collections_filter, **kwargs)
        assert client.request(method, path).json()["parents"] is None

    def test_parent_without_a_filter_is_skipped(self):
        """A parent path that no filter covers has nothing to check."""

        async def collections_filter(context):
            raise AssertionError("not called")

        client = _build_app(
            collections_filter,
            collections_filter_path=r"^/other$",
            sub_resource_endpoints=SUB_RESOURCES,
        )
        assert client.get("/collections/c1/queryables").json()["parents"] == []

    def test_filter_error_is_returned(self):
        """An HTTPException from the filter factory becomes the response."""

        async def collections_filter(context):
            raise HTTPException(status_code=403, detail="nope")

        client = _build_app(collections_filter, sub_resource_endpoints=SUB_RESOURCES)
        response = client.get("/collections/c1/queryables")
        assert response.status_code == 403
        assert response.json() == {"detail": "nope"}


def _validate_app(parent_filters) -> TestClient:
    """Build an app behind the validation middleware with fixed parent filters."""
    app = FastAPI()
    app.add_middleware(
        Cql2ValidateParentRecordsMiddleware, upstream_url="http://upstream"
    )

    @app.middleware("http")
    async def set_parents(request, call_next):
        if parent_filters is not None:
            request.state.cql2_parent_filters = parent_filters
        return await call_next(request)

    @app.get("/{path:path}")
    async def sub_resource():
        return {"served": True}

    return TestClient(app)


PRIVATE_FALSE = [("/collections/c1", Expr("private = false"))]


class TestValidateParentRecords:
    """Cql2ValidateParentRecordsMiddleware checks each parent record."""

    @pytest.mark.parametrize(
        "parent,expected_status",
        [
            pytest.param({"id": "c1", "private": False}, 200, id="readable"),
            pytest.param({"id": "c1", "private": True}, 404, id="hidden"),
            pytest.param(None, 404, id="missing"),
            pytest.param({"id": "c1"}, 404, id="cannot-evaluate"),
        ],
    )
    def test_parent_is_checked(self, parent, expected_status):
        """The sub-resource is served only when the parent matches its filter."""
        client = _validate_app(PRIVATE_FALSE)
        with patch.object(
            Cql2ValidateParentRecordsMiddleware,
            "_fetch_parent",
            new_callable=AsyncMock,
            return_value=parent,
        ) as fetch:
            response = client.get("/collections/c1/queryables")
        fetch.assert_awaited_once_with("/collections/c1")
        assert response.status_code == expected_status
        if expected_status == 404:
            assert response.json() == {
                "code": "NotFoundError",
                "description": "Record not found.",
            }

    def test_every_parent_must_match(self):
        """With several parents, one hidden parent hides the sub-resource."""
        parents = [
            ("/catalogs/cat1", Expr("private = false")),
            ("/collections/c1", Expr("private = false")),
        ]
        records = {
            "/catalogs/cat1": {"id": "cat1", "private": True},
            "/collections/c1": {"id": "c1", "private": False},
        }
        client = _validate_app(parents)
        with patch.object(
            Cql2ValidateParentRecordsMiddleware,
            "_fetch_parent",
            new_callable=AsyncMock,
            side_effect=lambda path: records[path],
        ):
            response = client.get("/catalogs/cat1/collections/c1/items")
        assert response.status_code == 404

    def test_upstream_error_is_a_bad_gateway(self):
        """A parent that cannot be fetched answers 502, not the sub-resource."""
        client = _validate_app(PRIVATE_FALSE)
        with patch.object(
            Cql2ValidateParentRecordsMiddleware,
            "_fetch_parent",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("down"),
        ):
            response = client.get("/collections/c1/queryables")
        assert response.status_code == 502
        assert response.json()["code"] == "UpstreamError"

    def test_no_parent_filters_pass_through(self):
        """Requests without parent filters are not checked."""
        client = _validate_app(None)
        with patch.object(
            Cql2ValidateParentRecordsMiddleware,
            "_fetch_parent",
            new_callable=AsyncMock,
        ) as fetch:
            assert client.get("/collections/c1/queryables").status_code == 200
        fetch.assert_not_awaited()


app_factory = AppFactory(
    oidc_discovery_url="https://example-stac-api.com/.well-known/openid-configuration",
    default_public=True,
    collections_filter={
        "cls": "stac_auth_proxy.filters:Template",
        "args": ["{{ '(private = false)' if payload is none else true }}"],
    },
)
ITEMS = {
    r"^/collections/(?P<collection_id>[^/]+)/items$": ["/collections/{collection_id}"]
}


@pytest.mark.parametrize("is_authenticated", [True, False], ids=["auth", "anon"])
@pytest.mark.parametrize("private", [True, False], ids=["private", "public"])
def test_sub_resource_of_a_hidden_collection(
    source_api_server, source_api_responses, token_builder, is_authenticated, private
):
    """Through the proxy, a sub-resource of a collection the caller cannot read is a 404."""
    source_api_responses["/collections/{collection_id}"]["GET"] = {
        "id": "foo",
        "private": private,
    }
    app = app_factory(upstream_url=source_api_server, sub_resource_endpoints=ITEMS)
    headers = (
        {"Authorization": f"Bearer {token_builder({'sub': 'test-user'})}"}
        if is_authenticated
        else {}
    )
    response = TestClient(app, headers=headers).get("/collections/foo/items")
    hidden = private and not is_authenticated
    assert response.status_code == (404 if hidden else 200)


def test_sub_resources_are_not_checked_by_default(
    source_api_server, source_api_responses
):
    """Without SUB_RESOURCE_ENDPOINTS, sub-resources are proxied as before."""
    source_api_responses["/collections/{collection_id}"]["GET"] = {
        "id": "foo",
        "private": True,
    }
    app = app_factory(upstream_url=source_api_server)
    assert TestClient(app).get("/collections/foo/items").status_code == 200
