"""API 응답이 스키마를 지키는지 확인하는 테스트 (모델 없이 가짜 탐지기로 실행)

실행: 레포 루트에서  python -m pytest -q
"""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import src.api.main as api
from src.score.schema import Detection, DetectionResult, validate


class FakeDetector:
    def __init__(self, found=True, fail=False):
        self.found, self.fail = found, fail

    def load(self):
        pass

    def status(self):
        return {"loaded": True, "weights": "fake"}

    def detect(self, img):
        if self.fail:
            raise RuntimeError("boom")
        dets = [Detection([img.width * 0.4, img.height * 0.4, img.width * 0.5, img.height * 0.55], 0.62)] if self.found else []
        return DetectionResult(img.width, img.height, dets, inference_ms=1.0, model="fake.pt")


def png(w=320, h=200, color=(90, 120, 60)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    def make(**kw):
        monkeypatch.setattr(api, "detector", FakeDetector(**kw))
        return TestClient(api.app)
    return make


def test_found(client):
    r = client(found=True).post("/detect", files={"image": ("a.png", png(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    validate(body, "score_response")
    assert body["found"] is True and isinstance(body["total_score"], int)


def test_not_found(client):
    r = client(found=False).post("/detect", files={"image": ("a.png", png(), "image/png")})
    body = r.json()
    validate(body, "score_response")
    assert body["found"] is False and body["total_score"] is None


def test_elapsed_sec_passthrough(client):
    r = client().post("/detect", files={"image": ("a.png", png(), "image/png")}, data={"elapsed_sec": "12.5"})
    assert r.json()["elapsed_sec"] == 12.5


@pytest.mark.parametrize("kwargs,status,code", [
    ({"files": {"image": ("a.txt", b"hello", "text/plain")}}, 400, "INVALID_IMAGE"),
    ({"files": {"image": ("a.png", b"not really a png", "image/png")}}, 400, "INVALID_IMAGE"),
    ({}, 400, "INVALID_IMAGE"),
    ({"files": {"image": ("a.png", png(), "image/png")}, "data": {"elapsed_sec": "-3"}}, 400, "INVALID_PARAM"),
    ({"files": {"image": ("a.png", b"0" * (10 * 1024 * 1024 + 1), "image/png")}}, 413, "IMAGE_TOO_LARGE"),
])
def test_errors(client, kwargs, status, code):
    r = client().post("/detect", **kwargs)
    assert r.status_code == status
    validate(r.json(), "error_response")
    assert r.json()["error"]["code"] == code


def test_broken_image_with_ultralytics_loaded(client):
    """ultralytics 를 불러오면 Image.open 이 바뀌어 다른 예외가 날 수 있음 → 그래도 400"""
    pytest.importorskip("ultralytics")
    import ultralytics  # noqa: F401  (Image.open 패치 적용)
    r = client().post("/detect", files={"image": ("a.png", b"not really a png", "image/png")})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "INVALID_IMAGE"


def test_model_error(client):
    r = client(fail=True).post("/detect", files={"image": ("a.png", png(), "image/png")})
    assert r.status_code == 500
    assert r.json()["error"]["code"] == "MODEL_ERROR"


def test_status_and_home(client):
    c = client()
    assert c.get("/api/status").json()["status"] == "online"
    assert c.get("/").status_code in (200, 404)
