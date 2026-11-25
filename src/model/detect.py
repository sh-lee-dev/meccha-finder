"""
meccha finder - YOLO 탐지 모듈

사진을 받아 schemas/detection.schema.json 형식의 DetectionResult 를 돌려줍니다.

설정 (환경변수, 없으면 기본값):
    MECCHA_WEIGHTS     가중치 파일 경로        기본: yolo26n.pt (일반 사물용 기본 모델)
    MECCHA_CONF        찾았다/못 찾았다 기준값  기본: 0.25
    MECCHA_IMGSZ       추론 해상도             기본: 640
    MECCHA_ANY_CLASS   1 이면 모든 클래스를 캐릭터로 취급 (기본 모델로 흐름만 테스트할 때)

어떤 클래스를 캐릭터로 볼지:
    1) 클래스가 1개뿐인 모델 (우리가 학습한 best.pt) → 그 클래스. 이름이 달라도 됨 (예: '11-18')
    2) 클래스 이름에 'chameleon' 이 있으면 → 그 클래스만
    3) 그 밖의 모델 (기본 COCO 모델 등) → MECCHA_ANY_CLASS=1 일 때만 모든 클래스, 아니면 항상 못 찾음
"""
from __future__ import annotations

import os
import time
from typing import Any, Optional

from PIL import Image

from src.score.schema import CLASS_NAME, DEFAULT_CONF_THRESHOLD, Detection, DetectionResult


def _env_bool(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class Detector:
    def __init__(
        self,
        weights: Optional[str] = None,
        conf: Optional[float] = None,
        imgsz: Optional[int] = None,
        any_class: Optional[bool] = None,
    ):
        self.weights = weights or os.getenv("MECCHA_WEIGHTS", "yolo26n.pt")
        self.conf = float(conf if conf is not None else os.getenv("MECCHA_CONF", DEFAULT_CONF_THRESHOLD))
        self.imgsz = int(imgsz if imgsz is not None else os.getenv("MECCHA_IMGSZ", 640))
        self.any_class = any_class if any_class is not None else _env_bool("MECCHA_ANY_CLASS")
        self._model = None
        self._target_ids: Optional[set[int]] = None  # None = 모든 클래스
        self.target_mode = "not_loaded"
        self.load_error: Optional[str] = None

    # ---------- 모델 로드 ----------
    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        except Exception as e:  # 가중치 파일 없음, 다운로드 실패 등
            self.load_error = f"{type(e).__name__}: {e}"
            raise
        names: dict[int, str] = self._model.names
        if len(names) == 1:
            self._target_ids = {next(iter(names))}
            self.target_mode = "single_class"
        elif CLASS_NAME in names.values():
            self._target_ids = {i for i, n in names.items() if n == CLASS_NAME}
            self.target_mode = "chameleon_class"
        elif self.any_class:
            self._target_ids = None
            self.target_mode = "any_class (테스트용)"
        else:
            self._target_ids = set()
            self.target_mode = "no_target (캐릭터 클래스가 없는 모델)"

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "weights": self.weights,
            "conf_threshold": self.conf,
            "imgsz": self.imgsz,
            "target_mode": self.target_mode,
            "classes": list(self._model.names.values())[:10] if self.loaded else None,
            "error": self.load_error,
        }

    # ---------- 탐지 ----------
    def detect(self, image: Image.Image) -> DetectionResult:
        """image: RGB PIL 이미지 (원본 크기). 박스는 원본 기준 픽셀로 돌려줌."""
        self.load()
        start = time.perf_counter()
        result = self._model.predict(source=image, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        inference_ms = round((time.perf_counter() - start) * 1000, 1)

        detections: list[Detection] = []
        if self._target_ids != set():
            for box in result.boxes:
                cls_id = int(box.cls.item())
                if self._target_ids is not None and cls_id not in self._target_ids:
                    continue
                x1, y1, x2, y2 = (max(0.0, round(float(v), 1)) for v in box.xyxy[0].tolist())
                detections.append(Detection(bbox=[x1, y1, x2, y2], confidence=round(float(box.conf.item()), 4)))
        detections.sort(key=lambda d: d.confidence, reverse=True)

        return DetectionResult(
            width=image.width,
            height=image.height,
            detections=detections,
            conf_threshold=self.conf,
            inference_ms=inference_ms,
            model=os.path.basename(self.weights),
        )
