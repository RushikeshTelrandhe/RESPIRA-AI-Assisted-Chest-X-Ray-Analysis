"""Safe file storage helpers. Randomized names, no path leaks to clients."""

from __future__ import annotations

import base64
import io
import uuid
from pathlib import Path

from PIL import Image

from backend.app.config import STORAGE_ROOT


def _decode_data_url(data_url: str) -> Image.Image:
    raw = data_url.split(",", 1)[1] if "," in data_url else data_url
    return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")


def save_upload_temp(doctor_id: str, filename: str, raw: bytes) -> Path:
    safe = "".join(c for c in filename if c.isalnum() or c in ("-", "_", "."))[-80:] or "upload.png"
    path = STORAGE_ROOT / "temp" / f"{doctor_id[:8]}-{uuid.uuid4().hex}-{safe}"
    path.write_bytes(raw)
    return path


def save_analysis_images(analysis_id: str, image: Image.Image, grad: dict | None, vit: dict | None) -> dict[str, str]:
    """Persist original + overlays. Returns paths RELATIVE to STORAGE_ROOT."""
    folder = STORAGE_ROOT / "analyses" / analysis_id
    folder.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}

    def _put(name: str, img: Image.Image) -> str:
        p = folder / name
        img.convert("RGB").save(p, format="PNG")
        return p.relative_to(STORAGE_ROOT).as_posix()

    out["original"] = _put("original.png", image.convert("RGB"))
    if grad:
        try:
            out["gradcam"] = _put("gradcam_heatmap.png", _decode_data_url(grad["heatmap"]))
            out["gradcam_overlay"] = _put("gradcam_overlay.png", _decode_data_url(grad["overlay"]))
        except Exception:
            pass
    if vit:
        try:
            out["vit"] = _put("vit_heatmap.png", _decode_data_url(vit["heatmap"]))
            out["vit_overlay"] = _put("vit_overlay.png", _decode_data_url(vit["overlay"]))
        except Exception:
            pass
    return out


def resolve(relative: str | Path | None) -> Path | None:
    if not relative:
        return None
    p = Path(relative)
    # Handle legacy rows that stored an absolute path: use as-is.
    if p.is_absolute():
        return p if p.exists() else None
    # Normal case: path relative to STORAGE_ROOT. Strip leading slashes
    # so "/reports/x.pdf" and "reports/x.pdf" both work.
    cleaned = str(relative).lstrip("/\\")
    if not cleaned:
        return None
    p = STORAGE_ROOT / cleaned
    return p if p.exists() else None
