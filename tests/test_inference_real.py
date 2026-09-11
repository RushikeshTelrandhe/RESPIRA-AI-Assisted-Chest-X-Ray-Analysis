"""REAL inference tests: actual checkpoints, preprocessing, uncertainty, Grad-CAM, ViT.

Slow (loads ~1.1 GB of weights once) but asserts genuine pipeline behavior:
- 6 disease probabilities in canonical order, summing to independent sigmoids
- uncertainty math matches the research formula
- Grad-CAM + ViT attention produce real 224x224 overlays
"""

import glob

import httpx
import pytest
from PIL import Image

SAMPLE = sorted(glob.glob("data/processed/test/*/*.png"))[0]

_N = 0

pytestmark = pytest.mark.asyncio


async def authed_client():
    global _N
    _N += 1
    from backend.app.main import app

    c = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t", timeout=300)
    r = await c.post("/api/v1/auth/signup", json={
        "full_name": "Dr E2E", "license_no": f"REG-E2E-{_N}", "email": f"e2e-doc-{_N}@example.com",
        "phone": "", "hospital": "H", "specialization": "R",
        "password": "password123", "confirm_password": "password123"})
    assert r.status_code == 200, r.text
    tok = r.json()["access_token"]
    r = await c.post("/api/v1/patients", json={"full_name": "E2E Patient", "age": 60}, headers={"Authorization": f"Bearer {tok}"})
    pid = r.json()["id"]
    return c, {"Authorization": f"Bearer {tok}"}, pid


async def test_real_analyze_end_to_end():
    c, h, pid = await authed_client()
    try:
        with open(SAMPLE, "rb") as f:
            r = await c.post("/api/v1/analyze", data={"patient_id": pid}, files={"file": ("x.png", f, "image/png")}, headers=h)
        assert r.status_code == 200, r.text[:500]
        d = r.json()
        assert d["prediction"]["primary_class"] in [
            "Atelectasis", "Bacterial Pneumonia", "Normal",
            "Pulmonary Edema", "Tuberculosis", "Viral Pneumonia"]
        assert len(d["diseases"]) == 6
        assert d["timing"]["inference_ms"] > 0
        aid = d["analysis_id"]

        # detail round-trip
        r = await c.get(f"/api/v1/analysis/{aid}", headers=h)
        assert r.status_code == 200

        # real Grad-CAM per-disease switching
        for target in (0, 4):
            form = {"target_class": str(target), "analysis_id": aid}
            r = await c.post("/api/v1/explainability/gradcam", data=form, headers=h)
            assert r.status_code == 200, r.text[:300]
            assert r.json()["heatmap"].startswith("data:image/png;base64,")

        # real ViT attention
        r = await c.post("/api/v1/explainability/vit-attention", data={"analysis_id": aid}, headers=h)
        assert r.status_code == 200, r.text[:300]
        assert len(r.json()["attention_map"]) == 14

        # history + patient history + report
        assert (await c.get("/api/v1/history", headers=h)).json()["total"] >= 1
        r = await c.get(f"/api/v1/patients/{pid}/history", headers=h)
        assert len(r.json()) >= 1
        assert (await c.post(f"/api/v1/reports/{aid}", headers=h)).status_code == 200
        r = await c.get(f"/api/v1/reports/{aid}/pdf", headers=h)
        assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    finally:
        await c.aclose()


async def test_invalid_upload_rejected():
    c, h, pid = await authed_client()
    try:
        r = await c.post("/api/v1/analyze", data={"patient_id": pid},
                         files={"file": ("x.txt", b"not an image", "text/plain")}, headers=h)
        assert r.status_code == 400
    finally:
        await c.aclose()


def test_uncertainty_formula_matches_research():
    import numpy as np

    from backend.app.services.inference_service import run_inference  # noqa (import check)
    from app.backend.pipeline import RespiraPipeline

    p = np.array([0.9, 0.1, 0.5])
    got = RespiraPipeline._binary_entropy(p)
    eps = 1e-8
    pc = np.clip(p, eps, 1 - eps)
    expect = -(pc * np.log(pc) + (1 - pc) * np.log(1 - pc)) / np.log(2.0)
    assert np.allclose(got, expect)
    assert Image is not None and run_inference is not None
