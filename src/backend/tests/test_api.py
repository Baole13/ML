"""API-level tests exercising the FastAPI routes via TestClient."""

from __future__ import annotations

import io

from scripts.make_synthetic_artifacts import _make_species_image


def _jpeg_bytes(species_idx: int, variant: int = 0) -> bytes:
    img = _make_species_image(species_idx, variant)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["encoder"] == "mock"


def test_model_info(client):
    resp = client.get("/api/v1/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["encoder"] == "mock"
    assert body["embedding_dim"] == 384
    assert body["num_classes"] == 5
    assert body["gallery_size"] > 0


def test_search_returns_expected_schema(client):
    files = {"image": ("q.jpg", _jpeg_bytes(0, 0), "image/jpeg")}
    resp = client.post("/api/v1/search", files=files, data={"k": 3})
    assert resp.status_code == 200, resp.text

    body = resp.json()
    assert set(body.keys()) >= {"prediction", "results", "model", "is_unknown"}

    pred = body["prediction"]
    assert set(pred.keys()) == {"species", "confidence"}
    assert pred["species"] in {"tiger", "wolf", "panda", "eagle", "dolphin"}
    assert 0.0 <= pred["confidence"] <= 1.0

    assert isinstance(body["results"], list)
    assert len(body["results"]) == 3
    for r in body["results"]:
        assert set(r.keys()) == {"id", "species", "similarity", "image_url"}
        assert r["image_url"].startswith("/gallery/")

    assert body["model"]["encoder"] == "mock"
    assert "version" in body["model"]
    assert body["is_unknown"] is None


def test_search_default_k(client):
    files = {"image": ("q.jpg", _jpeg_bytes(2, 1), "image/jpeg")}
    resp = client.post("/api/v1/search", files=files)
    assert resp.status_code == 200
    # Default k=5 per the route Form default.
    assert len(resp.json()["results"]) == 5


def test_search_rejects_non_image(client):
    files = {"image": ("bad.txt", b"hello world", "text/plain")}
    resp = client.post("/api/v1/search", files=files, data={"k": 2})
    assert resp.status_code == 400
    assert "detail" in resp.json()


def test_search_requires_image_field(client):
    resp = client.post("/api/v1/search", data={"k": 2})
    assert resp.status_code == 422  # missing required multipart field


def test_search_k_clamped_by_server(client):
    files = {"image": ("q.jpg", _jpeg_bytes(1, 0), "image/jpeg")}
    resp = client.post("/api/v1/search", files=files, data={"k": 100000})
    assert resp.status_code == 200
    gallery = resp.json()
    # Server caps k at MAX_K (20) and at gallery size.
    assert len(gallery["results"]) <= 20


def test_gallery_image_served(client):
    # First get a real filename from a search result.
    files = {"image": ("q.jpg", _jpeg_bytes(0, 0), "image/jpeg")}
    resp = client.post("/api/v1/search", files=files, data={"k": 1})
    url = resp.json()["results"][0]["image_url"]

    img_resp = client.get(url)
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"].startswith("image/")


def test_gallery_path_traversal_blocked(client):
    resp = client.get("/gallery/../../manifest.json")
    # Either 404 (normalized away) — never leaks the file.
    assert resp.status_code in (404, 307)
