"""End-to-end smoke test: run the real FastAPI app in-process and hit it over
HTTP exactly the way the frontend does (multipart upload + relative gallery
URLs). This avoids flaky background-process management and proves the
FE <-> backend contract works against the live app.

Run:  python scripts/e2e_smoke.py
"""

import io
import sys
import threading
import time
from pathlib import Path

import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os

os.environ["ENCODER"] = "mock"
os.environ["ARTIFACTS_DIR"] = str(ROOT / "_smoke_artifacts")

# 1) Build synthetic artifacts first.
from scripts.make_synthetic_artifacts import make_synthetic_artifacts, _make_species_image

make_synthetic_artifacts(ROOT / "_smoke_artifacts")

# 2) Start the real app in a background thread.
from app.main import app

config = uvicorn.Config(app, host="127.0.0.1", port=8137, log_level="warning")
server = uvicorn.Server(config)
thread = threading.Thread(target=server.run, daemon=True)
thread.start()

# Wait for startup (lifespan loads the engine).
base = "http://127.0.0.1:8137"
for _ in range(50):
    try:
        httpx.get(f"{base}/health", timeout=1.0)
        break
    except Exception:
        time.sleep(0.3)

ok = True
with httpx.Client(timeout=30.0, base_url=base) as client:
    # Health
    h = client.get("/health").json()
    print("HEALTH:", h)
    ok &= h.get("status") == "ok" and h.get("model_loaded") is True

    # Model info
    mi = client.get("/api/v1/model-info").json()
    print("MODEL-INFO:", {k: mi[k] for k in ("encoder", "num_classes", "gallery_size")})
    ok &= mi["encoder"] == "mock"

    # Search — exactly what services/api.ts sends (multipart image + k)
    buf = io.BytesIO()
    _make_species_image(2, 0).save(buf, format="JPEG", quality=92)
    files = {"image": ("query.jpg", buf.getvalue(), "image/jpeg")}
    r = client.post("/api/v1/search", files=files, data={"k": "4"})
    body = r.json()
    print("SEARCH status:", r.status_code)
    print("  prediction:", body["prediction"])
    print("  #results:", len(body["results"]), "| top species:", body["results"][0]["species"])
    ok &= r.status_code == 200
    ok &= body["prediction"]["species"] == "panda"
    ok &= len(body["results"]) == 4
    ok &= body["is_unknown"] is None

    # Follow the relative gallery URL the FE would put in <img src>
    img_url = body["results"][0]["image_url"]
    gi = client.get(img_url)
    print("GALLERY", img_url, "->", gi.status_code, gi.headers.get("content-type"))
    ok &= gi.status_code == 200 and gi.headers["content-type"].startswith("image/")

server.should_exit = True
thread.join(timeout=5)

print("\nE2E RESULT:", "PASS ✅" if ok else "FAIL ❌")
sys.exit(0 if ok else 1)
