"""
meccha finder - API 서버 (FastAPI)

사진 → [model] detect() → [score] evaluate() → JSON (schemas/score_response.schema.json)
frontend/app.py(승우)의 구조를 바탕으로, 응답을 스키마에 맞추고 백엔드 폴더로 옮긴 버전입니다.

실행 (레포 루트에서):
    python -m uvicorn src.api.main:app --reload --port 8000
    → 화면: http://127.0.0.1:8000   API 문서: http://127.0.0.1:8000/docs

설정 (환경변수): MECCHA_WEIGHTS, MECCHA_CONF, MECCHA_IMGSZ, MECCHA_ANY_CLASS  (src/model/detect.py 참고)
                MECCHA_CORS_ORIGINS  허용할 프론트 주소, 쉼표로 구분 (기본: * = 모두 허용)
"""
from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageOps

from src.model.detect import Detector
from src.score.evaluate import evaluate
from src.score.schema import SCHEMA_VERSION, error_response

log = logging.getLogger("meccha")
ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 10 * 1024 * 1024  # 10MB (프론트 MAX_MB 와 같게)
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

detector = Detector()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 서버 시작 때 모델을 미리 올리고 한 번 돌려 둠 (첫 요청이 느려지지 않게)
    try:
        detector.load()
        detector.detect(Image.new("RGB", (64, 64)))
        log.info("model ready: %s", detector.status())
    except Exception as e:
        log.error("model load failed: %s", e)  # 서버는 켜 두고, /detect 에서 MODEL_ERROR 로 응답
    yield


app = FastAPI(title="Meccha Finder API", version=SCHEMA_VERSION, lifespan=lifespan)

_origins = [o.strip() for o in os.getenv("MECCHA_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"], allow_headers=["*"])


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content=error_response(code, message))


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    fields = {str(e.get("loc", ["", ""])[-1]) for e in exc.errors()}
    if "image" in fields:
        return _error(400, "INVALID_IMAGE", "사진이 없어요. 사진을 골라서 다시 올려 주세요.")
    return _error(400, "INVALID_PARAM", f"요청 값이 올바르지 않아요: {', '.join(sorted(fields))}")


@app.get("/api/status")
async def status():
    """서버·모델 상태 (프론트 상단 표시나 점검용)"""
    return {"status": "online", "schema_version": SCHEMA_VERSION, "model": detector.status()}


@app.post("/detect")
@app.post("/api/detect", include_in_schema=False)  # frontend/app.py 와 같은 주소도 지원
async def detect(
    image: UploadFile = File(...),
    elapsed_sec: Optional[float] = Form(None, ge=0),
):
    # 1. 입력 검사
    data = await image.read()
    if len(data) > MAX_BYTES:
        return _error(413, "IMAGE_TOO_LARGE", "사진이 너무 커요. 10MB 이하로 올려 주세요.")
    if image.content_type and image.content_type not in ALLOWED_TYPES:
        return _error(400, "INVALID_IMAGE", "JPG, PNG, WEBP 사진만 올릴 수 있어요.")
    try:
        pil = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        # 사진이 아닌 파일은 PIL 오류 외에도 다른 예외가 날 수 있음
        # (예: ultralytics 가 Image.open 을 바꿔서 HEIF 로 다시 열려다 ModuleNotFoundError)
        return _error(400, "INVALID_IMAGE", "사진을 읽을 수 없어요. JPG, PNG, WEBP 파일로 다시 올려 주세요.")

    # 2. 탐지
    try:
        det = detector.detect(pil)
    except Exception:
        log.exception("detect failed")
        return _error(500, "MODEL_ERROR", "AI가 사진을 분석하지 못했어요. 잠시 후 다시 시도해 주세요.")

    # 3. 점수 계산
    try:
        result = evaluate(det, np.asarray(pil), elapsed_sec=elapsed_sec)
    except Exception:
        log.exception("evaluate failed")
        return _error(500, "INTERNAL_ERROR", "점수를 계산하지 못했어요. 잠시 후 다시 시도해 주세요.")
    return result.to_dict()


@app.get("/", response_class=HTMLResponse)
async def home():
    """프론트 화면 (frontend/index.html) - 서버 하나로 화면과 API를 같이 띄울 때"""
    page = ROOT / "frontend" / "index.html"
    if page.is_file():
        return HTMLResponse(page.read_text(encoding="utf-8"))
    return HTMLResponse("frontend/index.html 을 찾을 수 없어요.", status_code=404)
