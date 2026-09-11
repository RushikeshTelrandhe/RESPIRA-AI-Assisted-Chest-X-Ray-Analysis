"""Auth + patient authorization tests (no model weights required)."""

import httpx
import pytest

from backend.app.main import app

UID = 0


def fresh_email() -> str:
    global UID
    UID += 1
    return f"doc{UID}@pytest.example.com"


async def make_client():
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def signup(c: httpx.AsyncClient, email: str):
    r = await c.post("/api/v1/auth/signup", json={
        "full_name": "Dr Pytest", "license_no": "REG-PY", "email": email,
        "phone": "", "hospital": "H", "specialization": "Radiology",
        "password": "password123", "confirm_password": "password123"})
    assert r.status_code == 200, r.text
    return r.json()


@pytest.mark.asyncio
async def test_signup_login_me():
    async with await make_client() as c:
        email = fresh_email()
        body = await signup(c, email)
        assert body["access_token"]
        h = {"Authorization": f"Bearer {body['access_token']}"}
        r = await c.get("/api/v1/auth/me", headers=h)
        assert r.status_code == 200 and r.json()["email"] == email
        r = await c.post("/api/v1/auth/login", json={"email": email, "password": "password123"})
        assert r.status_code == 200
        r = await c.post("/api/v1/auth/login", json={"email": email, "password": "wrongpass1"})
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_patient_isolation_between_doctors():
    async with await make_client() as c:
        a = await signup(c, fresh_email())
        b = await signup(c, fresh_email())
        ha = {"Authorization": f"Bearer {a['access_token']}"}
        hb = {"Authorization": f"Bearer {b['access_token']}"}
        r = await c.post("/api/v1/patients", json={"full_name": "Alice"}, headers=ha)
        pid = r.json()["id"]
        # doctor B must not see / modify doctor A's patient
        assert (await c.get(f"/api/v1/patients/{pid}", headers=hb)).status_code == 404
        assert (await c.put(f"/api/v1/patients/{pid}", json={"notes": "x"}, headers=hb)).status_code == 404
        assert (await getattr(c, "delete")(f"/api/v1/patients/{pid}", headers=hb)).status_code == 404
        # doctor A can
        assert (await c.get(f"/api/v1/patients/{pid}", headers=ha)).status_code == 200
        # unauthenticated blocked
        assert (await c.get("/api/v1/patients")).status_code == 401
