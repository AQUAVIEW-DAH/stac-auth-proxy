"""
A token's issuer is checked against the issuer of the OIDC discovery document.

RFC 8725 (JWT Best Current Practices), section 3.8: a recipient must validate that the
cryptographic keys used to sign the JWT belong to the issuer, and RFC 9068, section 4: the
resource server must validate that the `iss` claim is the issuer identifier of its
authorization server, as its metadata names it (RFC 8414). The proxy already reads that
metadata, the discovery document, for its signing keys.
"""

import pytest
from conftest import MOCK_ISSUER
from fastapi.testclient import TestClient
from utils import AppFactory

from stac_auth_proxy.middleware.EnforceAuthMiddleware import OidcFetchError, OidcService

app_factory = AppFactory(
    oidc_discovery_url="https://example-stac-api.com/.well-known/openid-configuration",
    default_public=False,
    public_endpoints={},
    private_endpoints={},
)


def get_collections(app, token: str):
    """GET /collections with the token as a Bearer credential."""
    return TestClient(app).get(
        "/collections", headers={"Authorization": f"Bearer {token}"}
    )


def test_a_token_from_the_discovery_issuer_is_accepted(
    source_api_server, token_builder
):
    """A token whose iss is the issuer of the discovery document passes."""
    response = get_collections(
        app_factory(upstream_url=source_api_server), token_builder({})
    )
    assert response.status_code == 200


@pytest.mark.parametrize(
    "iss",
    [
        "https://another-issuer.example.com",
        MOCK_ISSUER + "/",  # issuers compare as exact strings
        MOCK_ISSUER.upper(),
    ],
)
def test_a_token_from_another_issuer_is_401(source_api_server, token_builder, iss):
    """A token signed with a trusted key but naming another issuer is refused."""
    response = get_collections(
        app_factory(upstream_url=source_api_server), token_builder({"iss": iss})
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid token issuer"
    assert response.headers.get("WWW-Authenticate", "").startswith("Bearer")


def test_a_token_without_an_issuer_is_401(source_api_server, token_builder):
    """A token with no iss cannot be checked, so it is refused."""
    response = get_collections(
        app_factory(upstream_url=source_api_server), token_builder({}, issuer=None)
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid token issuer"


def test_the_issuer_comes_from_the_document_not_from_the_url_it_was_read_from(
    source_api_server, token_builder
):
    """The discovery document may be read over an internal URL while it names the public issuer."""
    app = app_factory(
        upstream_url=source_api_server,
        oidc_discovery_internal_url="http://keycloak:8080/.well-known/openid-configuration",
    )
    assert get_collections(app, token_builder({})).status_code == 200
    internal = token_builder({"iss": "http://keycloak:8080"})
    assert get_collections(app, internal).status_code == 401


def test_a_discovery_document_without_an_issuer_is_refused(mock_jwks):
    """Without an issuer no token can be checked, so the proxy trusts none (fail closed)."""
    config = mock_jwks.return_value.json.return_value
    mock_jwks.return_value.json.return_value = {
        k: v for k, v in config.items() if k != "issuer"
    }
    with pytest.raises(OidcFetchError, match="issuer"):
        OidcService(
            oidc_discovery_url="https://example.com/.well-known/openid-configuration"
        )
