import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import os
    os.environ["ADMIN_EMAIL"], os.environ["ADMIN_PASSWORD"] = "admin@test.com", "Admin12345"
    from src.config import get_settings
    get_settings.cache_clear()
    from src.api import app
    return TestClient(app)


def _signup(c, email, pw="Passw0rd1"):
    return c.post("/auth/signup", json={"name": "T", "email": email, "password": pw, "confirm_password": pw})


def _token(c, email, pw="Passw0rd1"):
    return {"Authorization": "Bearer " + c.post("/auth/login", json={"email": email, "password": pw}).json()["token"]}


def test_signup_validation_and_duplicates(client):
    assert _signup(client, "a@test.com").status_code == 201
    assert _signup(client, "a@test.com").status_code == 409
    assert _signup(client, "b@test.com", "short").status_code == 400
    assert client.post("/auth/signup", json={"name": "T", "email": "c@test.com", "password": "Passw0rd1", "confirm_password": "Different1"}).status_code == 400


def test_login_failure_and_unauthenticated_access(client):
    assert client.post("/auth/login", json={"email": "a@test.com", "password": "wrong"}).status_code == 401
    assert client.post("/chat", json={"question": "hi"}).status_code == 401


def test_role_based_access_enforced_on_backend(client):
    assert client.get("/admin/users", headers=_token(client, "a@test.com")).status_code == 403
    assert client.get("/admin/users", headers=_token(client, "admin@test.com", "Admin12345")).status_code == 200


def test_conversation_isolation_between_users(client):
    _signup(client, "u1@test.com"), _signup(client, "u2@test.com")
    cid = client.post("/chat", json={"question": "How long is the AquaPearl 500 Pro warranty?"}, headers=_token(client, "u1@test.com")).json()["conversation_id"]
    assert client.get(f"/conversations/{cid}", headers=_token(client, "u2@test.com")).status_code == 404
    assert client.delete(f"/conversations/{cid}", headers=_token(client, "u2@test.com")).status_code == 404
    assert client.get(f"/conversations/{cid}", headers=_token(client, "u1@test.com")).status_code == 200


def test_customer_cannot_upload_documents(client):
    r = client.post("/admin/documents", files={"file": ("x.md", b"hello")}, headers=_token(client, "a@test.com"))
    assert r.status_code == 403
