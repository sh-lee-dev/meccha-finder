"""
meccha finder - 점수 모듈 데이터 스키마 (v0.1)

모델 -> 점수 -> UI 로 주고받는 데이터의 모양을 Python 타입으로 정의합니다.
JSON 형식의 기준은 schemas/*.schema.json 이고, 이 파일은 그 내용을 코드에서 쓰기 쉽게 옮긴 것입니다.
둘이 달라지면 JSON Schema 쪽이 기준입니다.

    from src.score.schema import Detection, DetectionResult, ScoreResponse, Ability
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

SCHEMA_VERSION = "0.1"
CLASS_NAME = "chameleon"

DEFAULT_CONF_THRESHOLD = 0.25  # 이 값 이상이면 "찾았다"

# 능력치 항목: 순서 = 차트에 그리는 순서.
# key 는 고정(코드용), label 은 임시 이름. 화면 이름은 프론트가 key 기준으로 정한다.
ABILITY_LABELS: dict[str, str] = {
    "evasion": "탐지 회피",        # 1 - confidence
    "color_blend": "색 동화",      # 박스 안 vs 주변 띠 색 분포 유사도
    "texture_blend": "질감 동화",  # 박스 안 vs 주변 띠 엣지·무늬 유사도
    "exposure": "노출 크기",       # 사진 대비 박스 면적이 작을수록 높음
    "clutter": "배경 복잡도",      # 주변 띠의 엣지 밀도가 높을수록 높음
    # "find_difficulty": "발견 난이도",  # 해상도별 발견 난이도, 검증 후 추가 (육각형)
}

SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


# ---------- 모델 -> 점수 ----------
@dataclass
class Detection:
    bbox: list[float]          # [x1, y1, x2, y2] 원본 사진 기준 픽셀
    confidence: float          # 0~1
    class_name: str = CLASS_NAME


@dataclass
class DetectionResult:
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)  # 신뢰도 높은 순, 못 찾으면 []
    conf_threshold: float = DEFAULT_CONF_THRESHOLD
    inference_ms: Optional[float] = None
    model: Optional[str] = None

    @property
    def found(self) -> bool:
        return len(self.detections) > 0

    @property
    def best(self) -> Optional[Detection]:
        return max(self.detections, key=lambda d: d.confidence) if self.detections else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "image": {"width": self.width, "height": self.height},
            "detections": [asdict(d) for d in self.detections],
            "conf_threshold": self.conf_threshold,
            "inference_ms": self.inference_ms,
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DetectionResult":
        return cls(
            width=d["image"]["width"],
            height=d["image"]["height"],
            detections=[Detection(**x) for x in d.get("detections", [])],
            conf_threshold=d.get("conf_threshold", DEFAULT_CONF_THRESHOLD),
            inference_ms=d.get("inference_ms"),
            model=d.get("model"),
        )


# ---------- 점수 -> UI ----------
@dataclass
class Ability:
    key: str
    score: Optional[int]       # 0~100, 계산 못 하면 None
    comment: str = ""

    @property
    def label(self) -> str:
        return ABILITY_LABELS[self.key]

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "score": self.score, "comment": self.comment}


@dataclass
class ScoreResponse:
    width: int
    height: int
    detection: Optional[Detection]   # 결과 화면에 그릴 대표 탐지 1개, 못 찾으면 None
    total_score: Optional[int]       # 0~100, 높을수록 잘 숨음. 못 찾으면 None (판정 불가)
    title: Optional[str]             # 임시 칭호. 못 찾으면 None
    abilities: list[Ability]
    elapsed_sec: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "found": self.detection is not None,
            "image": {"width": self.width, "height": self.height},
            "detection": (
                {"bbox": self.detection.bbox, "confidence": self.detection.confidence}
                if self.detection else None
            ),
            "total_score": self.total_score,
            "title": self.title,
            "abilities": [a.to_dict() for a in self.abilities],
            "elapsed_sec": self.elapsed_sec,
            "meta": self.meta,
        }

    @classmethod
    def not_found(cls, width: int, height: int, meta: Optional[dict[str, Any]] = None) -> "ScoreResponse":
        """못 찾았을 때: 점수·칭호·능력치를 모두 None 으로 (보여주는 방식은 프론트가 정함)."""
        return cls(width, height, None, None, None, [Ability(k, None) for k in ABILITY_LABELS], meta=meta or {})

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def error_response(code: str, message: str) -> dict[str, Any]:
    """POST /detect 실패 응답. code: INVALID_IMAGE | IMAGE_TOO_LARGE | INVALID_PARAM | MODEL_ERROR | INTERNAL_ERROR"""
    return {"error": {"code": code, "message": message}}


# ---------- 검증 (선택: pip install jsonschema) ----------
def validate(data: dict[str, Any], schema_name: str) -> None:
    """schema_name: 'detection' | 'score_response' | 'error_response'. 틀리면 예외 발생."""
    import jsonschema  # 개발·테스트 때만 필요

    schema = json.loads((SCHEMA_DIR / f"{schema_name}.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(data, schema)
