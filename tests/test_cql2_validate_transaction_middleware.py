"""Test Cql2ValidateTransactionMiddleware."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from cql2 import Expr
from fastapi import FastAPI, Request
from starlette.testclient import TestClient

from stac_auth_proxy.middleware.Cql2ValidateTransactionMiddleware import (
    Cql2ValidateTransactionMiddleware,
    _deep_merge,
)

ITEM_FILTER = {"op": "=", "args": [{"property": "collection"}, "allowed"]}
COLLECTION_FILTER = {"op": "=", "args": [{"property": "id"}, "my-collection"]}


@pytest.fixture
def cql2_filter():
    """Return a CQL2 filter that matches collection = 'allowed'."""
    return Expr(ITEM_FILTER)


@pytest.fixture
def app_with_middleware():
    """Create a FastAPI app with the transaction middleware."""

    def _create(upstream_url="http://upstream:8080"):
        app = FastAPI()
        app.add_middleware(
            Cql2ValidateTransactionMiddleware,
            upstream_url=upstream_url,
        )

        @app.post("/collections/{collection_id}/items")
        async def create_item(request: Request):
            body = await request.body()
            return json.loads(body) if body else {}

        @app.put("/collections/{collection_id}/items/{item_id}")
        async def update_item_put(request: Request):
            body = await request.body()
            return json.loads(body) if body else {}

        @app.patch("/collections/{collection_id}/items/{item_id}")
        async def update_item_patch(request: Request):
            body = await request.body()
            return json.loads(body) if body else {}

        @app.delete("/collections/{collection_id}/items/{item_id}")
        async def delete_item(request: Request):
            return {"deleted": True}

        @app.post("/collections/{collection_id}/bulk_items")
        async def bulk_create_items(request: Request):
            body = await request.body()
            return json.loads(body) if body else {}

        @app.post("/collections")
        async def create_collection(request: Request):
            body = await request.body()
            return json.loads(body) if body else {}

        @app.put("/collections/{collection_id}")
        async def update_collection_put(request: Request):
            body = await request.body()
            return json.loads(body) if body else {}

        @app.patch("/collections/{collection_id}")
        async def update_collection_patch(request: Request):
            body = await request.body()
            return json.loads(body) if body else {}

        @app.delete("/collections/{collection_id}")
        async def delete_collection(request: Request):
            return {"deleted": True}

        @app.post("/catalogs")
        @app.put("/catalogs/{catalog_id}")
        @app.post("/catalogs/{catalog_id}/{children}")
        @app.put("/catalogs/{catalog_id}/collections/{collection_id}")
        async def catalog_write(request: Request):
            body = await request.body()
            return json.loads(body) if body else {}

        @app.delete("/catalogs/{catalog_id}")
        @app.delete("/catalogs/{catalog_id}/{children}/{child_id}")
        async def catalog_delete(request: Request):
            return {"deleted": True}

        @app.get("/search")
        async def search_get(request: Request):
            return {"type": "FeatureCollection", "features": []}

        @app.post("/search")
        async def search_post(request: Request):
            return {"type": "FeatureCollection", "features": []}

        @app.get("/collections/{collection_id}/items/{item_id}")
        async def get_item(request: Request):
            return {"id": "item1", "collection": "allowed"}

        return app

    return _create


def _set_cql2_filter(app, cql2_filter, cql2_read_filter=None):
    """Add middleware that sets cql2_filter (and cql2_read_filter) on request state."""

    @app.middleware("http")
    async def set_filter(request, call_next):
        request.state.cql2_filter = cql2_filter
        if cql2_read_filter is not None:
            request.state.cql2_read_filter = cql2_read_filter
        return await call_next(request)


class TestDeepMerge:
    """Test the _deep_merge utility function."""

    @pytest.mark.parametrize(
        "base,override,expected",
        [
            pytest.param({"a": 1}, {"b": 2}, {"a": 1, "b": 2}, id="disjoint-keys"),
            pytest.param({"a": 1}, {"a": 2}, {"a": 2}, id="override-value"),
            pytest.param(
                {"properties": {"name": "old", "count": 1}},
                {"properties": {"name": "new"}},
                {"properties": {"name": "new", "count": 1}},
                id="nested-dict",
            ),
            pytest.param(
                {"a": {"b": {"c": 1, "d": 2}}},
                {"a": {"b": {"c": 3}}},
                {"a": {"b": {"c": 3, "d": 2}}},
                id="deeply-nested",
            ),
            pytest.param(
                {"a": {"nested": True}},
                {"a": "replaced"},
                {"a": "replaced"},
                id="dict-to-non-dict",
            ),
            pytest.param(
                {"a": "string"},
                {"a": {"nested": True}},
                {"a": {"nested": True}},
                id="non-dict-to-dict",
            ),
        ],
    )
    def test_merge(self, base, override, expected):
        """Deep merge produces expected result."""
        assert _deep_merge(base, override) == expected


class TestCreate:
    """Test item and collection creation validation."""

    @pytest.mark.parametrize(
        "path,body,filter_expr,expected_status",
        [
            pytest.param(
                "/collections/test/items",
                {"id": "item1", "collection": "allowed"},
                ITEM_FILTER,
                200,
                id="item-allowed",
            ),
            pytest.param(
                "/collections/test/items",
                {"id": "item1", "collection": "denied"},
                ITEM_FILTER,
                403,
                id="item-denied",
            ),
            pytest.param(
                "/collections",
                {"id": "my-collection", "type": "Collection"},
                COLLECTION_FILTER,
                200,
                id="collection-allowed",
            ),
            pytest.param(
                "/collections",
                {"id": "denied-collection", "type": "Collection"},
                COLLECTION_FILTER,
                403,
                id="collection-denied",
            ),
        ],
    )
    def test_create(
        self, app_with_middleware, path, body, filter_expr, expected_status
    ):
        """Allow or deny creation based on whether body matches filter."""
        app = app_with_middleware()
        _set_cql2_filter(app, Expr(filter_expr))
        client = TestClient(app)
        response = client.post(path, json=body)
        assert response.status_code == expected_status
        if expected_status == 403:
            assert response.json()["code"] == "ForbiddenError"

    def test_create_no_filter(self, app_with_middleware):
        """Request passes through when no CQL2 filter is set."""
        app = app_with_middleware()
        client = TestClient(app)
        response = client.post(
            "/collections/test/items",
            json={"id": "item1", "collection": "anything"},
        )
        assert response.status_code == 200


class TestBulkCreate:
    """Test bulk item creation validation."""

    def test_all_items_allowed(self, app_with_middleware, cql2_filter):
        """All items match filter, request passes through."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter)
        client = TestClient(app)
        response = client.post(
            "/collections/test/bulk_items",
            json={
                "items": {
                    "item1": {"id": "item1", "collection": "allowed"},
                    "item2": {"id": "item2", "collection": "allowed"},
                }
            },
        )
        assert response.status_code == 200

    def test_some_items_denied(self, app_with_middleware, cql2_filter):
        """Some items fail filter, returns 403 with failed item IDs."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter)
        client = TestClient(app)
        response = client.post(
            "/collections/test/bulk_items",
            json={
                "items": {
                    "item1": {"id": "item1", "collection": "allowed"},
                    "item2": {"id": "item2", "collection": "denied"},
                }
            },
        )
        assert response.status_code == 403
        body = response.json()
        assert body["code"] == "ForbiddenError"
        assert "item2" in body["description"]

    def test_all_items_denied(self, app_with_middleware, cql2_filter):
        """All items fail filter, returns 403."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter)
        client = TestClient(app)
        response = client.post(
            "/collections/test/bulk_items",
            json={
                "items": {
                    "item1": {"id": "item1", "collection": "denied"},
                    "item2": {"id": "item2", "collection": "denied"},
                }
            },
        )
        assert response.status_code == 403
        body = response.json()
        assert body["code"] == "ForbiddenError"
        assert "item1" in body["description"]
        assert "item2" in body["description"]

    def test_no_filter(self, app_with_middleware):
        """Request passes through when no CQL2 filter is set."""
        app = app_with_middleware()
        client = TestClient(app)
        response = client.post(
            "/collections/test/bulk_items",
            json={
                "items": {
                    "item1": {"id": "item1", "collection": "anything"},
                }
            },
        )
        assert response.status_code == 200

    def test_empty_items(self, app_with_middleware, cql2_filter):
        """Empty items dict passes through."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter)
        client = TestClient(app)
        response = client.post(
            "/collections/test/bulk_items",
            json={"items": {}},
        )
        assert response.status_code == 200

    def test_invalid_json(self, app_with_middleware, cql2_filter):
        """Invalid JSON returns 400."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter)
        client = TestClient(app)
        response = client.post(
            "/collections/test/bulk_items",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "ParseError"

    def test_items_not_object(self, app_with_middleware, cql2_filter):
        """Items field that is not a dict returns 400."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter)
        client = TestClient(app)
        response = client.post(
            "/collections/test/bulk_items",
            json={"items": [{"id": "item1", "collection": "allowed"}]},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "ParseError"


class TestItemCollectionCreate:
    """Test validation of an ItemCollection posted to the items endpoint."""

    @staticmethod
    def _item_collection(*features):
        return {"type": "FeatureCollection", "features": list(features)}

    def test_all_items_allowed(self, app_with_middleware, cql2_filter):
        """All items match filter, request passes through."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter)
        client = TestClient(app)
        body = self._item_collection(
            {"type": "Feature", "id": "item1", "collection": "allowed"},
            {"type": "Feature", "id": "item2", "collection": "allowed"},
        )
        response = client.post("/collections/allowed/items", json=body)
        assert response.status_code == 200
        assert response.json() == body

    def test_some_items_denied(self, app_with_middleware):
        """Some items fail filter, returns 403 with failed item IDs."""
        app = app_with_middleware()
        _set_cql2_filter(app, Expr({"op": "=", "args": [{"property": "id"}, "item1"]}))
        client = TestClient(app)
        response = client.post(
            "/collections/allowed/items",
            json=self._item_collection(
                {"type": "Feature", "id": "item1"},
                {"type": "Feature", "id": "item2"},
            ),
        )
        assert response.status_code == 403
        body = response.json()
        assert body["code"] == "ForbiddenError"
        assert "item2" in body["description"]
        assert "item1" not in body["description"]

    @pytest.mark.parametrize(
        "path,collection,expected_status",
        [
            pytest.param("/collections/allowed/items", None, 200, id="no-collection"),
            pytest.param(
                "/collections/denied/items", "allowed", 403, id="other-collection"
            ),
        ],
    )
    def test_items_are_checked_in_the_collection_of_the_path(
        self, app_with_middleware, cql2_filter, path, collection, expected_status
    ):
        """The server populates each item's collection from the path, so the check does too."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter)
        client = TestClient(app)
        feature = {"type": "Feature", "id": "item1"}
        if collection:
            feature["collection"] = collection
        response = client.post(path, json=self._item_collection(feature))
        assert response.status_code == expected_status

    def test_no_filter(self, app_with_middleware):
        """Request passes through when no CQL2 filter is set."""
        app = app_with_middleware()
        client = TestClient(app)
        response = client.post(
            "/collections/test/items",
            json=self._item_collection({"type": "Feature", "id": "item1"}),
        )
        assert response.status_code == 200

    def test_empty_features(self, app_with_middleware, cql2_filter):
        """An ItemCollection without features passes through."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter)
        client = TestClient(app)
        response = client.post(
            "/collections/allowed/items", json=self._item_collection()
        )
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "features",
        [
            pytest.param({"item1": {"id": "item1"}}, id="features-not-array"),
            pytest.param(["item1"], id="feature-not-object"),
        ],
    )
    def test_malformed_features(self, app_with_middleware, cql2_filter, features):
        """Features that are not an array of objects return 400."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter)
        client = TestClient(app)
        response = client.post(
            "/collections/allowed/items",
            json={"type": "FeatureCollection", "features": features},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "ParseError"


class TestUpdate:
    """Test item and collection update validation."""

    @pytest.mark.parametrize(
        "path,existing,body,filter_expr,expected_status,error_code",
        [
            pytest.param(
                "/collections/allowed/items/item1",
                {"id": "item1", "collection": "allowed", "properties": {"name": "old"}},
                {"id": "item1", "collection": "allowed", "properties": {"name": "new"}},
                ITEM_FILTER,
                200,
                None,
                id="item-allowed",
            ),
            pytest.param(
                "/collections/denied/items/item1",
                {"id": "item1", "collection": "denied", "properties": {}},
                {"id": "item1", "collection": "allowed"},
                ITEM_FILTER,
                404,
                "NotFoundError",
                id="item-existing-not-found",
            ),
            pytest.param(
                "/collections/allowed/items/item1",
                {"id": "item1", "collection": "allowed", "properties": {}},
                {"id": "item1", "collection": "denied", "properties": {}},
                ITEM_FILTER,
                403,
                "ForbiddenError",
                id="item-result-denied",
            ),
            pytest.param(
                "/collections/my-collection",
                {"id": "my-collection", "type": "Collection"},
                {"id": "my-collection", "type": "Collection", "title": "Updated"},
                COLLECTION_FILTER,
                200,
                None,
                id="collection-allowed",
            ),
        ],
    )
    def test_put(
        self,
        app_with_middleware,
        path,
        existing,
        body,
        filter_expr,
        expected_status,
        error_code,
    ):
        """PUT validates both existing record and new body against filter."""
        app = app_with_middleware()
        _set_cql2_filter(app, Expr(filter_expr))
        client = TestClient(app)
        with patch.object(
            Cql2ValidateTransactionMiddleware,
            "_fetch_existing",
            new_callable=AsyncMock,
            return_value=existing,
        ):
            response = client.put(path, json=body)
        assert response.status_code == expected_status
        if error_code:
            assert response.json()["code"] == error_code

    @pytest.mark.parametrize(
        "existing,body,expected_status",
        [
            pytest.param(
                {"id": "item1", "collection": "allowed", "properties": {"name": "old"}},
                {"properties": {"name": "new"}},
                200,
                id="allowed",
            ),
            pytest.param(
                {
                    "id": "item1",
                    "collection": "allowed",
                    "properties": {"name": "old", "count": 5},
                    "assets": {"thumbnail": {"href": "http://example.com/thumb.png"}},
                },
                {"properties": {"name": "new"}},
                200,
                id="merge-preserves-collection",
            ),
            pytest.param(
                {"id": "item1", "collection": "allowed", "properties": {}},
                {"collection": "denied"},
                403,
                id="changes-collection-denied",
            ),
        ],
    )
    def test_patch(
        self, app_with_middleware, cql2_filter, existing, body, expected_status
    ):
        """PATCH merges body with existing record and validates the result."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter)
        client = TestClient(app)
        with patch.object(
            Cql2ValidateTransactionMiddleware,
            "_fetch_existing",
            new_callable=AsyncMock,
            return_value=existing,
        ):
            response = client.patch("/collections/allowed/items/item1", json=body)
        assert response.status_code == expected_status
        if expected_status == 403:
            assert response.json()["code"] == "ForbiddenError"


class TestDelete:
    """Test item and collection deletion validation."""

    @pytest.mark.parametrize(
        "path,existing,filter_expr,expected_status,error_code",
        [
            pytest.param(
                "/collections/allowed/items/item1",
                {"id": "item1", "collection": "allowed"},
                ITEM_FILTER,
                200,
                None,
                id="item-allowed",
            ),
            pytest.param(
                "/collections/denied/items/item1",
                {"id": "item1", "collection": "denied"},
                ITEM_FILTER,
                404,
                "NotFoundError",
                id="item-denied",
            ),
            pytest.param(
                "/collections/my-collection",
                {"id": "my-collection", "type": "Collection"},
                COLLECTION_FILTER,
                200,
                None,
                id="collection-allowed",
            ),
            pytest.param(
                "/collections/other-collection",
                {"id": "other-collection", "type": "Collection"},
                COLLECTION_FILTER,
                404,
                "NotFoundError",
                id="collection-denied",
            ),
        ],
    )
    def test_delete(
        self,
        app_with_middleware,
        path,
        existing,
        filter_expr,
        expected_status,
        error_code,
    ):
        """Delete validates existing record against filter."""
        app = app_with_middleware()
        _set_cql2_filter(app, Expr(filter_expr))
        client = TestClient(app)
        with patch.object(
            Cql2ValidateTransactionMiddleware,
            "_fetch_existing",
            new_callable=AsyncMock,
            return_value=existing,
        ):
            response = client.delete(path)
        assert response.status_code == expected_status
        if error_code:
            assert response.json()["code"] == error_code


class TestCatalogs:
    """Test Multi-Tenant Catalogs transaction validation."""

    OWNER_FILTER = {"op": "=", "args": [{"property": "owner"}, "me"]}
    RECORDS = {
        "/catalogs/mine": {"id": "mine", "type": "Catalog", "owner": "me"},
        "/catalogs/theirs": {"id": "theirs", "type": "Catalog", "owner": "them"},
        "/catalogs/my-sub": {"id": "my-sub", "type": "Catalog", "owner": "me"},
        "/catalogs/their-sub": {"id": "their-sub", "type": "Catalog", "owner": "them"},
        "/collections/my-col": {"id": "my-col", "type": "Collection", "owner": "me"},
        "/collections/their-col": {
            "id": "their-col",
            "type": "Collection",
            "owner": "them",
        },
        "/catalogs/mine/collections/my-col": {
            "id": "my-col",
            "type": "Collection",
            "owner": "me",
        },
        "/catalogs/mine/collections/their-col": {
            "id": "their-col",
            "type": "Collection",
            "owner": "them",
        },
    }

    def _request(self, app_with_middleware, method, path, **kwargs):
        """Send a request with the owner filter; return the response and fetched paths."""
        app = app_with_middleware()
        _set_cql2_filter(app, Expr(self.OWNER_FILTER))
        client = TestClient(app)
        with patch.object(
            Cql2ValidateTransactionMiddleware,
            "_fetch_existing",
            new_callable=AsyncMock,
            side_effect=self.RECORDS.get,
        ) as fetch:
            response = client.request(method, path, **kwargs)
        return response, [call.args[0] for call in fetch.await_args_list]

    @pytest.mark.parametrize(
        "method,path,body,expected_status,error_code",
        [
            pytest.param(
                "POST",
                "/catalogs",
                {"id": "new", "type": "Catalog", "owner": "me"},
                200,
                None,
                id="create-allowed",
            ),
            pytest.param(
                "POST",
                "/catalogs",
                {"id": "new", "type": "Catalog", "owner": "them"},
                403,
                "ForbiddenError",
                id="create-denied",
            ),
            pytest.param(
                "PUT",
                "/catalogs/mine",
                {"id": "mine", "type": "Catalog", "owner": "me", "title": "New"},
                200,
                None,
                id="update-allowed",
            ),
            pytest.param(
                "PUT",
                "/catalogs/theirs",
                {"id": "theirs", "type": "Catalog", "owner": "me"},
                404,
                "NotFoundError",
                id="update-existing-denied",
            ),
            pytest.param(
                "PUT",
                "/catalogs/mine",
                {"id": "mine", "type": "Catalog", "owner": "them"},
                403,
                "ForbiddenError",
                id="update-result-denied",
            ),
            pytest.param(
                "DELETE", "/catalogs/mine", None, 200, None, id="delete-allowed"
            ),
            pytest.param(
                "DELETE",
                "/catalogs/theirs",
                None,
                404,
                "NotFoundError",
                id="delete-denied",
            ),
            pytest.param(
                "PUT",
                "/catalogs/mine/collections/my-col",
                {"id": "my-col", "type": "Collection", "owner": "me"},
                200,
                None,
                id="scoped-collection-update-allowed",
            ),
            pytest.param(
                "PUT",
                "/catalogs/mine/collections/their-col",
                {"id": "their-col", "type": "Collection", "owner": "me"},
                404,
                "NotFoundError",
                id="scoped-collection-update-denied",
            ),
        ],
    )
    def test_catalog_records(
        self, app_with_middleware, method, path, body, expected_status, error_code
    ):
        """Catalogs are created, updated, and deleted like collections."""
        response, _ = self._request(app_with_middleware, method, path, json=body)
        assert response.status_code == expected_status
        if error_code:
            assert response.json()["code"] == error_code

    @pytest.mark.parametrize(
        "path,body,expected_status,error_code,fetched",
        [
            pytest.param(
                "/catalogs/mine/collections",
                {"id": "new-col", "type": "Collection", "owner": "me"},
                200,
                None,
                ["/catalogs/mine", "/collections/new-col"],
                id="create-collection",
            ),
            pytest.param(
                "/catalogs/mine/collections",
                {"id": "new-col", "type": "Collection", "owner": "them"},
                403,
                "ForbiddenError",
                ["/catalogs/mine", "/collections/new-col"],
                id="create-collection-denied",
            ),
            pytest.param(
                "/catalogs/theirs/collections",
                {"id": "new-col", "type": "Collection", "owner": "me"},
                404,
                "NotFoundError",
                ["/catalogs/theirs"],
                id="create-collection-in-denied-catalog",
            ),
            pytest.param(
                "/catalogs/missing/collections",
                {"id": "new-col", "type": "Collection", "owner": "me"},
                404,
                "NotFoundError",
                ["/catalogs/missing"],
                id="create-collection-in-missing-catalog",
            ),
            pytest.param(
                "/catalogs/mine/collections",
                {"id": "my-col"},
                200,
                None,
                ["/catalogs/mine", "/collections/my-col"],
                id="link-collection",
            ),
            pytest.param(
                "/catalogs/mine/collections",
                {"id": "their-col"},
                403,
                "ForbiddenError",
                ["/catalogs/mine", "/collections/their-col"],
                id="link-collection-denied",
            ),
            pytest.param(
                "/catalogs/mine/collections",
                {"id": "their-col", "owner": "me"},
                403,
                "ForbiddenError",
                ["/catalogs/mine", "/collections/their-col"],
                id="link-collection-denied-whatever-the-body-says",
            ),
            pytest.param(
                "/catalogs/theirs/collections",
                {"id": "my-col"},
                404,
                "NotFoundError",
                ["/catalogs/theirs"],
                id="link-collection-into-denied-catalog",
            ),
            pytest.param(
                "/catalogs/mine/catalogs",
                {"id": "new-sub", "type": "Catalog", "owner": "me"},
                200,
                None,
                ["/catalogs/mine", "/catalogs/new-sub"],
                id="create-sub-catalog",
            ),
            pytest.param(
                "/catalogs/mine/catalogs",
                {"id": "my-sub"},
                200,
                None,
                ["/catalogs/mine", "/catalogs/my-sub"],
                id="link-sub-catalog",
            ),
            pytest.param(
                "/catalogs/mine/collections",
                {"id": "a b?c", "owner": "them"},
                403,
                "ForbiddenError",
                ["/catalogs/mine", "/collections/a%20b%3Fc"],
                id="child-id-is-quoted",
            ),
        ],
    )
    def test_add_child(
        self,
        app_with_middleware,
        path,
        body,
        expected_status,
        error_code,
        fetched,
    ):
        """Adding a child checks the catalog, and the linked record or the new body."""
        response, paths = self._request(app_with_middleware, "POST", path, json=body)
        assert response.status_code == expected_status
        if error_code:
            assert response.json()["code"] == error_code
        assert paths == fetched

    @pytest.mark.parametrize(
        "body",
        [
            pytest.param({"type": "Collection", "owner": "me"}, id="no-id"),
            pytest.param({"id": 1, "owner": "me"}, id="id-not-a-string"),
            pytest.param({"id": "a/b", "owner": "me"}, id="id-with-slash"),
            pytest.param([{"id": "my-col"}], id="not-an-object"),
        ],
    )
    def test_add_child_needs_an_id(self, app_with_middleware, body):
        """A child that cannot be looked up by its id is refused."""
        response, paths = self._request(
            app_with_middleware, "POST", "/catalogs/mine/collections", json=body
        )
        assert response.status_code == 400
        assert response.json()["code"] == "ParseError"
        assert paths == []

    def test_add_child_invalid_json(self, app_with_middleware):
        """Invalid JSON returns 400."""
        response, paths = self._request(
            app_with_middleware,
            "POST",
            "/catalogs/mine/collections",
            content=b"not json",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "ParseError"
        assert paths == []

    @pytest.mark.parametrize(
        "path,expected_status,fetched",
        [
            pytest.param(
                "/catalogs/mine/collections/my-col",
                200,
                ["/collections/my-col", "/catalogs/mine"],
                id="unlink-collection",
            ),
            pytest.param(
                "/catalogs/mine/collections/their-col",
                404,
                ["/collections/their-col"],
                id="unlink-denied-collection",
            ),
            pytest.param(
                "/catalogs/theirs/collections/my-col",
                404,
                ["/collections/my-col", "/catalogs/theirs"],
                id="unlink-from-denied-catalog",
            ),
            pytest.param(
                "/catalogs/mine/catalogs/my-sub",
                200,
                ["/catalogs/my-sub", "/catalogs/mine"],
                id="unlink-sub-catalog",
            ),
            pytest.param(
                "/catalogs/mine/catalogs/their-sub",
                404,
                ["/catalogs/their-sub"],
                id="unlink-denied-sub-catalog",
            ),
        ],
    )
    def test_remove_child(self, app_with_middleware, path, expected_status, fetched):
        """Unlinking checks the child, then the catalog."""
        response, paths = self._request(app_with_middleware, "DELETE", path)
        assert response.status_code == expected_status
        if expected_status == 404:
            assert response.json()["code"] == "NotFoundError"
        assert paths == fetched

    @pytest.mark.parametrize(
        "method,path,kwargs",
        [
            pytest.param(
                "POST",
                "/catalogs/mine/collections",
                {"json": {"id": "my-col"}},
                id="add-child",
            ),
            pytest.param(
                "DELETE", "/catalogs/mine/collections/my-col", {}, id="remove-child"
            ),
        ],
    )
    def test_upstream_unreachable(self, app_with_middleware, method, path, kwargs):
        """Returns 502 when a record cannot be fetched."""
        app = app_with_middleware()
        _set_cql2_filter(app, Expr(self.OWNER_FILTER))
        client = TestClient(app)
        with patch.object(
            Cql2ValidateTransactionMiddleware,
            "_fetch_existing",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            response = client.request(method, path, **kwargs)
        assert response.status_code == 502
        assert response.json()["code"] == "UpstreamError"

    def test_upstream_unreachable_for_the_child(self, app_with_middleware):
        """Returns 502 when the record to link cannot be fetched."""

        def fetch(path):
            if path == "/catalogs/mine":
                return self.RECORDS[path]
            raise httpx.ConnectError("Connection refused")

        app = app_with_middleware()
        _set_cql2_filter(app, Expr(self.OWNER_FILTER))
        client = TestClient(app)
        with patch.object(
            Cql2ValidateTransactionMiddleware,
            "_fetch_existing",
            new_callable=AsyncMock,
            side_effect=fetch,
        ):
            response = client.post("/catalogs/mine/collections", json={"id": "my-col"})
        assert response.status_code == 502
        assert response.json()["code"] == "UpstreamError"

    def test_no_filter(self, app_with_middleware):
        """Catalog writes pass through when no CQL2 filter is set."""
        app = app_with_middleware()
        client = TestClient(app)
        response = client.post("/catalogs/theirs/collections", json={"id": "their-col"})
        assert response.status_code == 200


class TestPassthrough:
    """Test that non-transaction requests pass through unmodified."""

    @pytest.mark.parametrize(
        "method,path,kwargs",
        [
            pytest.param("get", "/collections/test/items/item1", {}, id="get-item"),
            pytest.param(
                "post", "/search", {"json": {"collections": ["test"]}}, id="post-search"
            ),
            pytest.param("get", "/search", {}, id="get-search"),
        ],
    )
    def test_passthrough(self, app_with_middleware, cql2_filter, method, path, kwargs):
        """Non-transaction requests pass through without validation."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter)
        client = TestClient(app)
        response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 200


class TestUpstreamFetchFailure:
    """Test behavior when upstream is unreachable."""

    @pytest.mark.parametrize(
        "method,path,kwargs",
        [
            pytest.param(
                "put",
                "/collections/allowed/items/item1",
                {"json": {"id": "item1", "collection": "allowed"}},
                id="put",
            ),
            pytest.param("delete", "/collections/allowed/items/item1", {}, id="delete"),
        ],
    )
    def test_upstream_unreachable(
        self, app_with_middleware, cql2_filter, method, path, kwargs
    ):
        """Returns 502 when upstream fetch raises UpstreamError."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter)
        client = TestClient(app)
        with patch.object(
            Cql2ValidateTransactionMiddleware,
            "_fetch_existing",
            new_callable=AsyncMock,
            side_effect=httpx.ConnectError("Connection refused"),
        ):
            response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 502
        assert response.json()["code"] == "UpstreamError"

    @pytest.mark.parametrize(
        "method,path,kwargs",
        [
            pytest.param(
                "put",
                "/collections/allowed/items/item1",
                {"json": {"id": "item1", "collection": "allowed"}},
                id="put",
            ),
            pytest.param("delete", "/collections/allowed/items/item1", {}, id="delete"),
        ],
    )
    def test_record_not_found(
        self, app_with_middleware, cql2_filter, method, path, kwargs
    ):
        """Returns 404 when upstream record does not exist."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter)
        client = TestClient(app)
        with patch.object(
            Cql2ValidateTransactionMiddleware,
            "_fetch_existing",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = getattr(client, method)(path, **kwargs)
        assert response.status_code == 404
        assert response.json()["code"] == "NotFoundError"


class TestReadableButNotWritable:
    """A record the caller may read but not modify is refused with 403, not hidden."""

    READ_ALL = {"op": "isNull", "args": [{"property": "no_such_field"}]}
    READ_NONE = {"op": "=", "args": [{"property": "collection"}, "nothing"]}

    @pytest.mark.parametrize(
        "method,kwargs",
        [
            pytest.param(
                "put",
                {"json": {"id": "item1", "collection": "denied"}},
                id="put",
            ),
            pytest.param("patch", {"json": {"properties": {}}}, id="patch"),
            pytest.param("delete", {}, id="delete"),
        ],
    )
    @pytest.mark.parametrize(
        "read_filter,expected_status,error_code",
        [
            pytest.param(READ_ALL, 403, "ForbiddenError", id="readable"),
            pytest.param(READ_NONE, 404, "NotFoundError", id="not-readable"),
            pytest.param(None, 404, "NotFoundError", id="no-read-filter"),
        ],
    )
    def test_existing_record_denied_by_write_filter(
        self,
        app_with_middleware,
        cql2_filter,
        method,
        kwargs,
        read_filter,
        expected_status,
        error_code,
    ):
        """403 if the read filter matches the existing record, else 404."""
        app = app_with_middleware()
        _set_cql2_filter(
            app, cql2_filter, Expr(read_filter) if read_filter is not None else None
        )
        client = TestClient(app)
        with patch.object(
            Cql2ValidateTransactionMiddleware,
            "_fetch_existing",
            new_callable=AsyncMock,
            return_value={"id": "item1", "collection": "denied", "properties": {}},
        ):
            response = getattr(client, method)(
                "/collections/denied/items/item1", **kwargs
            )
        assert response.status_code == expected_status
        assert response.json()["code"] == error_code

    @pytest.mark.parametrize("method", ["put", "delete"])
    def test_read_filter_that_cannot_be_evaluated_is_404(
        self, app_with_middleware, cql2_filter, method
    ):
        """A read filter that raises on the record fails closed: 404, not 500."""
        app = app_with_middleware()
        # `IN` on a list-valued property cannot be reduced to a boolean
        _set_cql2_filter(app, cql2_filter, Expr("tags IN ('a', 'b')"))
        client = TestClient(app)
        with patch.object(
            Cql2ValidateTransactionMiddleware,
            "_fetch_existing",
            new_callable=AsyncMock,
            return_value={"id": "item1", "collection": "denied", "tags": ["a"]},
        ):
            response = getattr(client, method)(
                "/collections/denied/items/item1",
                **({"json": {"id": "item1"}} if method == "put" else {}),
            )
        assert response.status_code == 404
        assert response.json()["code"] == "NotFoundError"

    @pytest.mark.parametrize("method", ["put", "delete"])
    def test_missing_record_is_404_even_when_everything_is_readable(
        self, app_with_middleware, cql2_filter, method
    ):
        """A record that does not exist stays a 404."""
        app = app_with_middleware()
        _set_cql2_filter(app, cql2_filter, Expr(self.READ_ALL))
        client = TestClient(app)
        with patch.object(
            Cql2ValidateTransactionMiddleware,
            "_fetch_existing",
            new_callable=AsyncMock,
            return_value=None,
        ):
            response = getattr(client, method)(
                "/collections/denied/items/item1",
                **({"json": {"id": "item1"}} if method == "put" else {}),
            )
        assert response.status_code == 404
        assert response.json()["code"] == "NotFoundError"
