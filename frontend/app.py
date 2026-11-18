"""
메챠 파인더 - 프론트엔드 통합 서버 (FastAPI)

이 서버는 프론트엔드 UI(index.html)와 팀원들의 모듈(YOLO 탐지, 점수 알고리즘)을
안정적으로 이어주는 중계 허브 역할을 합니다.

주요 특징:
- YOLO 모듈(yolo_connector)과 점수 모듈(score_connector)이 완벽히 분리되어 있어,
  다른 팀원들의 코드가 완성되는 즉시 별도 코드 수정 없이 자동 연결됩니다.
- 모듈이 아직 준비되지 않았더라도 자체 Mock 모드로 프론트엔드 개발 및 시각화 테스트가 가능합니다.
- 프론트엔드에서 실시간으로 YOLO 연결 상태와 신뢰도(confidence)를 제어할 수 있는 API를 제공합니다.

실행: python app.py  →  브라우저에서 http://127.0.0.1:8000 접속
"""

import io
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# 현재 폴더 및 모듈 임포트
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from yolo_connector import yolo_connector
from score_connector import score_connector

app = FastAPI(
    title="Meccha Finder API",
    description="메챠 파인더 UI & YOLO & 점수 모듈 통합 백엔드 API",
    version="1.0.0",
)

# CORS 허용 (로컬 개발 및 Live Server 대응)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────────────────
# 1. 상태 확인 및 설정 API (프론트엔드 연동용)
# ────────────────────────────────────────────────────────────
@app.get("/api/status")
async def get_system_status():
    """
    프론트엔드가 페이지를 로드할 때 호출하여
    YOLO 연결 상태 및 점수 모듈 연결 상태를 화면 상단 배지에 표시합니다.
    """
    yolo_status = yolo_connector.get_status()
    score_status = score_connector.get_status()
    return {
        "status": "online",
        "yolo": yolo_status,
        "score": score_status,
    }


@app.post("/api/config")
async def update_config(
    mode: Optional[str] = Form(None),
    confidence: Optional[float] = Form(None),
    remote_url: Optional[str] = Form(None),
):
    """프론트엔드 설정 창에서 YOLO 모드나 confidence threshold를 동적으로 변경"""
    if confidence is not None:
        yolo_connector.set_confidence(confidence)
    if mode is not None:
        yolo_connector.set_mode(mode, remote_url=remote_url)
    return {
        "success": True,
        "current_status": yolo_connector.get_status(),
    }


# ────────────────────────────────────────────────────────────
# 2. 메인 탐지 & 점수 통합 API (POST /api/detect, POST /detect)
# ────────────────────────────────────────────────────────────
@app.post("/api/detect")
@app.post("/detect")
async def handle_detect(
    image: UploadFile = File(...),
    confidence: Optional[float] = Form(None),
):
    """
    프론트엔드에서 이미지를 전송받아:
    1. YOLO 커넥터로 객체 탐지 수행
    2. 점수 커넥터로 은폐 점수 및 능력치 산출
    3. 프론트엔드가 즉시 렌더링할 수 있는 완성된 데이터 반환
    """
    try:
        # 1. 이미지 로딩
        contents = await image.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # 신뢰도 임시 지정이 있을 경우 적용
        if confidence is not None:
            yolo_connector.set_confidence(confidence)

        # 2. YOLO 모듈 연결 및 탐지 수행
        detection_result = yolo_connector.detect(pil_img)

        # 3. 점수 모듈 연결 및 채점 수행
        elapsed_sec = detection_result.get("elapsed_sec", 0.0)
        score_result = score_connector.evaluate(detection_result, elapsed_sec=elapsed_sec)

        # 4. 프론트엔드 응답 조합 (이전 버전 호환성 + 신규 포맷 모두 제공)
        response_payload = {
            "success": True,
            "filename": image.filename,
            "model": detection_result.get("model_name", "yolo"),
            "mode_used": detection_result.get("mode_used", "auto"),
            "found": detection_result.get("found", False),
            "confidence": detection_result.get("confidence", 0.0),
            "bbox": detection_result.get("bbox"),
            "objects": detection_result.get("objects", []),
            "image_size": detection_result.get("image_size", [pil_img.width, pil_img.height]),
            "elapsed_sec": elapsed_sec,
            # 점수 및 능력치
            "total_score": score_result.get("total", 0),
            "title": score_result.get("title", ""),
            "abilities": score_result.get("axes", {}),
            "comments": score_result.get("comments", {}),
            "comment_list": score_result.get("comment_list", []),
        }

        return JSONResponse(content=response_payload)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"탐지 처리 중 오류 발생: {str(e)}"},
        )


# ────────────────────────────────────────────────────────────
# 3. HTML 프론트엔드 서빙
# ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_home():
    """index.html 웹 화면 서빙"""
    html_candidates = [
        BASE_DIR / "index.html",
        BASE_DIR.parent / "src" / "ui" / "index.html",
    ]
    for p in html_candidates:
        if p.is_file():
            return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("index.html 을 찾을 수 없습니다.", status_code=404)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 60)
    print("🦎 [메챠 파인더] 프론트엔드 통합 서버 시작")
    print("👉 브라우저 접속: http://127.0.0.1:8000")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
