"""Test the Allow header on non-preflight OPTIONS requests (ENABLE_OPTIONS_ALLOW)."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from cql2 import Expr
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from utils import AppFactory

from stac_auth_proxy.middleware.Cql2BuildFilterMiddleware import (
    Cql2BuildFilterMiddleware,
)
from stac_auth_proxy.middleware.OptionsAllowMiddleware import OptionsAllowMiddleware
from stac_auth_proxy.utils.requests import is_cors_preflight

# Anyone may read public records; a signed-in user may also read and change their own.
OWNER_FILTER = """
{%- if req.method == 'GET' -%}
  {{ "private = false" if payload is none else "private = false OR owner = '" ~ payload.sub ~ "'" }}
{%- else -%}
  owner = '{{ payload.sub if payload else "" }}'
{%- endif -%}
"""

PUBLIC_RECORD = {"id": "r1", "collection": "c1", "owner": "alice", "private": False}
PRIVATE_RECORD = {"id": "r1", "collection": "c1", "owner": "alice", "private": True}

SINGLE_RECORDS = [
    pytest.param("/collections/c1", "/collections/{collection_id}", id="collection"),
    pytest.param(
        "/collections/c1/items/r1",
        "/collections/{collection_id}/items/{item_id}",
        id="item",
    ),
]

app_factory = AppFactory(
    oidc_discovery_url="https://example-stac-api.com/.well-known/openid-configuration",
    default_public=True,
    enable_options_allow=True,
    collections_filter={
        "cls": "stac_auth_proxy.filters:Template",
        "args": [OWNER_FILTER.strip()],
    },
    items_filter={
        "cls": "stac_auth_proxy.filters:Template",
        "args": [OWNER_FILTER.strip()],
    },
)


@pytest.fixture
def client_for(source_api_server, token_builder):
    """Build a client for an anonymous caller, or for a signed-in user."""

    def _client(sub=None, scope="", **overrides):
        headers = (
            {"Authorization": f"Bearer {token_builder({'sub': sub, 'scope': scope})}"}
            if sub
            else {}
        )
        app = app_factory(upstream_url=source_api_server, **overrides)
        return TestClient(app, headers=headers)

    return _client


def allowed(response) -> list[str]:
    """Get the methods listed in the Allow header."""
    assert response.status_code == 200, response.text
    return response.headers["allow"].split(", ")


class TestSingleRecord:
    """On a single record, each method is checked against the record."""

    @pytest.mark.parametrize("path,route", SINGLE_RECORDS)
    def test_anonymous_caller_may_read(
        self, client_for, source_api_responses, path, route
    ):
        """No token: the public answer."""
        source_api_responses[route]["GET"] = PUBLIC_RECORD
        response = client_for().options(path)
        assert allowed(response) == ["GET", "HEAD", "OPTIONS"]
        assert response.content == b""

    @pytest.mark.parametrize("path,route", SINGLE_RECORDS)
    def test_owner_may_change_the_record(
        self, client_for, source_api_responses, path, route
    ):
        """The write filter matches the record: PUT, PATCH, and DELETE are listed."""
        source_api_responses[route]["GET"] = PUBLIC_RECORD
        response = client_for("alice").options(path)
        assert allowed(response) == [
            "GET",
            "HEAD",
            "PUT",
            "PATCH",
            "DELETE",
            "OPTIONS",
        ]

    @pytest.mark.parametrize("path,route", SINGLE_RECORDS)
    def test_reader_may_not_change_the_record(
        self, client_for, source_api_responses, path, route
    ):
        """The read filter matches and the write filter does not."""
        source_api_responses[route]["GET"] = PUBLIC_RECORD
        response = client_for("bob").options(path)
        assert allowed(response) == ["GET", "HEAD", "OPTIONS"]

    @pytest.mark.parametrize("sub", [None, "bob"], ids=["anonymous", "other-user"])
    @pytest.mark.parametrize("path,route", SINGLE_RECORDS)
    def test_record_the_caller_may_not_read_is_not_found(
        self, client_for, source_api_responses, path, route, sub
    ):
        """A hidden record is reported as not found, as a GET would be."""
        source_api_responses[route]["GET"] = PRIVATE_RECORD
        response = client_for(sub).options(path)
        assert response.status_code == 404
        assert response.json()["code"] == "NotFoundError"
        assert "allow" not in response.headers

    def test_owner_sees_their_private_record(self, client_for, source_api_responses):
        """The owner may read, and change, their private record."""
        source_api_responses["/collections/{collection_id}"]["GET"] = PRIVATE_RECORD
        response = client_for("alice").options("/collections/c1")
        assert "PUT" in allowed(response)

    def test_missing_record_is_not_found(self, client_for):
        """A missing record gets the same answer as a hidden one."""
        with patch.object(
            OptionsAllowMiddleware,
            "_fetch_record",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = client_for("alice").options("/collections/c1")
        assert response.status_code == 404
        assert response.json()["code"] == "NotFoundError"

    def test_upstream_failure(self, client_for):
        """The record cannot be fetched: 502."""
        with patch.object(
            OptionsAllowMiddleware,
            "_fetch_record",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("connection refused"),
        ):
            response = client_for("alice").options("/collections/c1")
        assert response.status_code == 502
        assert response.json()["code"] == "UpstreamError"

    def test_filter_that_cannot_be_evaluated_is_no_match(
        self, client_for, source_api_responses
    ):
        """A write filter that raises on the record leaves the write methods out."""
        source_api_responses["/collections/{collection_id}"]["GET"] = {
            **PUBLIC_RECORD,
            "tags": ["a"],
        }
        client = client_for(
            "alice",
            collections_filter={
                "cls": "stac_auth_proxy.filters:Template",
                # `IN` on a list-valued property cannot be reduced to a boolean
                "args": [
                    "{{ 'true' if req.method == 'GET' else \"tags IN ('a', 'b')\" }}"
                ],
            },
        )
        response = client.options("/collections/c1")
        assert allowed(response) == ["GET", "HEAD", "OPTIONS"]


class TestResourcesEndpoint:
    """On a list endpoint, POST is listed when the caller may create."""

    @pytest.mark.parametrize(
        "path",
        ["/collections", "/collections/c1/items"],
    )
    def test_anonymous_caller_may_not_create(self, client_for, path):
        """POST needs a token."""
        response = client_for().options(path)
        assert allowed(response) == ["GET", "HEAD", "OPTIONS"]

    @pytest.mark.parametrize(
        "path",
        ["/collections", "/collections/c1/items"],
    )
    def test_signed_in_caller_may_create(self, client_for, path):
        """The body of a create is checked on the request itself."""
        response = client_for("alice").options(path)
        assert allowed(response) == ["GET", "HEAD", "POST", "OPTIONS"]

    @pytest.mark.parametrize(
        "scope,expected",
        [
            pytest.param("", ["GET", "HEAD", "OPTIONS"], id="missing-scope"),
            pytest.param(
                "collection:create",
                ["GET", "HEAD", "POST", "OPTIONS"],
                id="with-scope",
            ),
        ],
    )
    def test_required_scopes(self, client_for, scope, expected):
        """A method that needs a scope is listed only for a token with that scope."""
        client = client_for(
            "alice",
            scope=scope,
            private_endpoints={r"^/collections$": [("POST", "collection:create")]},
        )
        assert allowed(client.options("/collections")) == expected

    def test_no_write_methods_where_no_writes_are_configured(self, client_for):
        """Write methods come from PRIVATE_ENDPOINTS."""
        response = client_for("alice").options("/search")
        assert allowed(response) == ["GET", "HEAD", "OPTIONS"]


class TestWithoutFilters:
    """Without a CQL2 filter, only the endpoint rules decide."""

    @pytest.mark.parametrize(
        "sub,expected",
        [
            pytest.param(None, ["OPTIONS"], id="anonymous"),
            pytest.param(
                "alice",
                ["GET", "HEAD", "PUT", "PATCH", "DELETE", "OPTIONS"],
                id="signed-in",
            ),
        ],
    )
    def test_endpoint_rules(self, client_for, sub, expected):
        """Every endpoint needs a token; no record is fetched."""
        client = client_for(
            sub,
            default_public=False,
            public_endpoints={},
            collections_filter=None,
            items_filter=None,
        )
        with patch.object(
            OptionsAllowMiddleware, "_fetch_record", new_callable=AsyncMock
        ) as fetch:
            response = client.options("/collections/c1")
        assert allowed(response) == expected
        fetch.assert_not_called()


class TestAuthentication:
    """The token is optional on OPTIONS, but a token that is sent must be valid."""

    @pytest.mark.parametrize(
        "auth_header",
        ["Bearer invalid-token", "InvalidFormat"],
    )
    def test_invalid_token_is_rejected(self, client_for, auth_header):
        """An invalid token gets 401, as on any other request."""
        response = client_for().options(
            "/collections", headers={"Authorization": auth_header}
        )
        assert response.status_code == 401

    def test_disabled_by_default(self, source_api_server):
        """Without the setting, OPTIONS requests are passed on as before."""
        app = app_factory(
            upstream_url=source_api_server,
            enable_options_allow=False,
            proxy_options=True,
        )
        response = TestClient(app).options(
            "/collections", headers={"Authorization": "Bearer invalid-token"}
        )
        assert response.status_code == 200
        assert response.json()["id"] == "Response from OPTIONS@"


class TestCors:
    """CORS preflight requests are not changed."""

    PREFLIGHT = {
        "Origin": "https://example.com",
        "Access-Control-Request-Method": "PUT",
        "Access-Control-Request-Headers": "Authorization",
    }

    def test_preflight_is_answered_by_cors_middleware(self, client_for):
        """A preflight gets the CORS answer, whatever the token."""
        response = client_for().options(
            "/collections/c1",
            headers={**self.PREFLIGHT, "Authorization": "Bearer invalid-token"},
        )
        assert response.status_code == 200
        assert response.text == "OK"
        assert response.headers["access-control-allow-origin"] == "https://example.com"

    def test_cross_origin_options_request(self, client_for, source_api_responses):
        """A cross-origin OPTIONS request gets CORS headers, and may read Allow."""
        source_api_responses["/collections/{collection_id}"]["GET"] = PUBLIC_RECORD
        response = client_for("alice").options(
            "/collections/c1", headers={"Origin": "https://example.com"}
        )
        assert "PUT" in allowed(response)
        assert response.headers["access-control-allow-origin"] == "https://example.com"
        assert "Allow" in response.headers["access-control-expose-headers"]

    def test_with_proxy_options(self, client_for, source_api_responses):
        """With PROXY_OPTIONS, preflights still go upstream."""
        source_api_responses["/collections/{collection_id}"]["GET"] = PUBLIC_RECORD
        client = client_for("alice", proxy_options=True)

        response = client.options("/collections/c1")
        assert "PUT" in allowed(response)

        response = client.options("/collections/c1", headers=self.PREFLIGHT)
        assert response.status_code == 200
        assert response.json()["id"] == "Response from OPTIONS@"


class TestFiltersForOptions:
    """Cql2BuildFilterMiddleware builds the filter for each method on OPTIONS requests."""

    PATH = "/collections/c1/items/r1"

    @staticmethod
    def _app(items_filter, options_filters=True):
        app = FastAPI()
        app.add_middleware(
            Cql2BuildFilterMiddleware,
            items_filter=items_filter,
            options_filters=options_filters,
        )

        @app.options("/collections/{collection_id}/items/{item_id}")
        async def endpoint(request: Request):
            filters = getattr(request.state, "cql2_filters", None)
            if filters is None:
                return {"filters": None}
            return {"filters": {m: f.to_text() for m, f in filters.items()}}

        return app

    @staticmethod
    async def _method_filter(context):
        return f"method = '{context['req']['method']}'"

    def test_filter_for_each_method(self):
        """The factory is called once per method, with that method."""
        client = TestClient(self._app(self._method_filter))
        response = client.options(self.PATH)
        assert response.json()["filters"] == {
            method: Expr(f"method = '{method}'").to_text()
            for method in ["GET", "POST", "PUT", "PATCH", "DELETE"]
        }

    def test_not_built_by_default(self):
        """Without the option, OPTIONS requests still skip filter building."""
        client = TestClient(self._app(self._method_filter, options_filters=False))
        assert client.options(self.PATH).json()["filters"] is None

    def test_not_built_for_preflight(self):
        """A CORS preflight request skips filter building."""
        client = TestClient(self._app(self._method_filter))
        response = client.options(
            self.PATH,
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "PUT",
            },
        )
        assert response.json()["filters"] is None

    def test_method_whose_filter_cannot_be_built_is_left_out(self):
        """A factory error leaves that method out."""

        async def items_filter(context):
            if context["req"]["method"] != "GET":
                raise HTTPException(status_code=403, detail="read only")
            return "collection = 'c1'"

        client = TestClient(self._app(items_filter))
        assert client.options(self.PATH).json()["filters"] == {
            "GET": Expr("collection = 'c1'").to_text()
        }


@pytest.mark.parametrize(
    "headers,expected",
    [
        pytest.param(
            {"Origin": "https://a.example", "Access-Control-Request-Method": "PUT"},
            True,
            id="preflight",
        ),
        pytest.param({"Origin": "https://a.example"}, False, id="no-request-method"),
        pytest.param({"Access-Control-Request-Method": "PUT"}, False, id="no-origin"),
        pytest.param({}, False, id="plain"),
    ],
)
def test_is_cors_preflight(headers, expected):
    """A preflight request carries Origin and Access-Control-Request-Method."""
    request = Request(
        {
            "type": "http",
            "method": "OPTIONS",
            "path": "/",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        }
    )
    assert is_cors_preflight(request) is expected
