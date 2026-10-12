"""Unit tests for the search engine and artifact loading."""

from __future__ import annotations

import numpy as np

from app.services.artifacts import ArtifactError, load_artifacts
from app.services.engine import build_engine
from app.services.preprocessing import (
    InvalidImageError,
    build_spec_from_manifest,
    load_pil_image,
    normalize_embeddings,
)


def test_artifacts_load_and_validate(engine):
    info = engine.info()
    assert info["gallery_size"] > 0
    assert info["num_classes"] == 5
    assert info["embedding_dim"] == 384
    assert info["metric"] == "cosine"
    assert info["index_type"] == "IndexFlatIP"


def test_predict_returns_known_species(engine, sample_image_bytes):
    pred = engine.predict(sample_image_bytes)
    assert pred.species in {"tiger", "wolf", "panda", "eagle", "dolphin"}
    assert 0.0 <= pred.confidence <= 1.0


def test_search_returns_topk(engine, sample_image_bytes):
    results = engine.search(sample_image_bytes, k=3)
    assert len(results) == 3
    for r in results:
        assert r.species in {"tiger", "wolf", "panda", "eagle", "dolphin"}
        assert r.image_url.startswith("/gallery/")
        # Cosine similarity of L2-normalized vectors is in [-1, 1].
        assert -1.0001 <= r.similarity <= 1.0001
    # Scores are sorted descending (FAISS returns best-first).
    sims = [r.similarity for r in results]
    assert sims == sorted(sims, reverse=True)


def test_search_k_clamped_to_gallery(engine, sample_image_bytes):
    results = engine.search(sample_image_bytes, k=9999)
    assert len(results) == engine.info()["gallery_size"]


def test_self_retrieval_hits_exact_match(engine, sample_image_bytes):
    # The uploaded image IS a gallery image, so top-1 should be itself with
    # cosine ~= 1.0.
    results = engine.search(sample_image_bytes, k=1)
    assert results[0].similarity > 0.99


def test_invalid_image_raises(engine):
    try:
        engine.predict(b"not-an-image")
    except InvalidImageError:
        return
    raise AssertionError("Expected InvalidImageError for non-image bytes")


def test_normalize_embeddings_unit_norm():
    x = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], dtype="float32")
    y = normalize_embeddings(x)
    assert abs(np.linalg.norm(y[0]) - 1.0) < 1e-5
    # Zero row must not blow up (guarded to norm 1).
    assert np.all(np.isfinite(y))


def test_preprocess_spec_from_manifest():
    spec = build_spec_from_manifest(
        {"image_size": 224, "normalization": "imagenet"}
    )
    assert spec.image_size == 224
    assert spec.mean == (0.485, 0.456, 0.406)


def test_missing_artifacts_raise(tmp_path):
    try:
        load_artifacts(tmp_path / "does-not-exist")
    except ArtifactError:
        return
    raise AssertionError("Expected ArtifactError for missing artifacts dir")


def test_build_engine_happy_path(artifacts_dir, monkeypatch):
    from app.config import Settings
    from app.services.engine import build_engine

    monkeypatch.setenv("ENCODER", "mock")
    s = Settings(artifacts_dir=artifacts_dir)
    s.encoder = "mock"
    eng = build_engine(s)
    assert eng.info()["gallery_size"] > 0
    assert eng.info()["embedding_dim"] == 384


def test_search_and_predict_encodes_once(engine, sample_image_bytes, monkeypatch):
    """The search endpoint must encode the image exactly once per request."""

    calls = {"n": 0}
    real_embed = engine._embed

    def counting_embed(data):
        calls["n"] += 1
        return real_embed(data)

    monkeypatch.setattr(engine, "_embed", counting_embed)
    outcome = engine.search_and_predict(sample_image_bytes, k=3)

    assert calls["n"] == 1, "encoder ran more than once per request"
    assert outcome.prediction.species
    assert len(outcome.results) == 3


def test_uncalibrated_classifier_rejected_when_required(artifacts_dir, monkeypatch):
    """Startup must fail fast if the head lacks predict_proba (production)."""

    import joblib
    from sklearn.svm import LinearSVC

    from app.config import Settings
    from app.services.artifacts import load_artifacts
    from app.services.engine import build_engine

    monkeypatch.setenv("ENCODER", "mock")

    clf_path = artifacts_dir / "dino_classifier.joblib"
    original = joblib.load(clf_path)  # calibrated LogisticRegression

    try:
        # Replace with a bare LinearSVC (no predict_proba).
        bundle = load_artifacts(artifacts_dir)
        svc = LinearSVC(C=1.0, max_iter=5000)
        svc.fit(bundle.embeddings, bundle.labels)
        joblib.dump(svc, clf_path)
        assert not hasattr(svc, "predict_proba")

        s = Settings(artifacts_dir=artifacts_dir)
        s.encoder = "mock"

        # Default (require_calibrated=True) -> hard failure.
        raised = False
        try:
            build_engine(s)
        except ValueError:
            raised = True
        assert raised, "Expected ValueError for uncalibrated classifier"

        # Opt-out path still works (softmax fallback), flagged calibrated=False.
        s.require_calibrated = False
        eng = build_engine(s)
        assert eng.info()["calibrated"] is False
    finally:
        # Restore the calibrated head so other (session-scoped) tests are safe.
        joblib.dump(original, clf_path)


