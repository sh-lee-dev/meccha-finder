"""
메챠 파인더 - 점수 커넥터

app.py 가 기대하는 인터페이스만 노출합니다.

    score_connector.get_status()
    score_connector.evaluate(detection, elapsed_sec=None, image=None)

detection 은 yolo_connector.detect() 가 돌려준 Detection 스키마 dict 입니다.
image 는 박스 주변의 색·질감을 재기 위한 PIL 이미지(선택)입니다.
이미지 없이 호출해도 되고, 그때는 픽셀이 필요한 항목만 null 이 됩니다.

반환값은 docs/score-schema.md 3장 / schemas/score_response.schema.json 과
완전히 같은 형식입니다.

    {
      "schema_version": "0.1",
      "found": true,
      "image": {"width": 1390, "height": 781},
      "detection": {"bbox": [412.0, 168.0, 538.0, 305.0], "confidence": 0.87},
      "total_score": 50,
      "title": "아쉬운 견습 카멜레온",
      "abilities": [
        {"key": "evasion", "label": "탐지 회피", "score": 13, "comment": "AI가 바로 찾았어요."},
        ...
      ],
      "elapsed_sec": null,
      "meta": {"model": "best_v1.pt", "inference_ms": 41.5, "conf_threshold": 0.25}
    }

- 능력치 축과 순서는 src/score/schema.py 의 ABILITY_LABELS 를 따릅니다.
- 총점(v0)은 null 이 아닌 항목의 평균(반올림)입니다. (가중치는 3주차 확정)
- 못 찾았을 때는 총점·칭호·능력치를 모두 null 로 보냅니다. (판정 불가)
- 계산은 Pillow 만으로 하므로 numpy/OpenCV 설치가 필요 없습니다.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Optional

try:
    from PIL import Image, ImageFilter, ImageStat  # noqa: F401
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageFilter = None  # type: ignore[assignment]
    ImageStat = None  # type: ignore[assignment]

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

# 팀 스키마(src/score/schema.py)를 우선 사용하고, 없으면 같은 내용을 내장
_FALLBACK_LABELS: dict[str, str] = {
    "evasion": "탐지 회피",
    "color_blend": "색 동화",
    "texture_blend": "질감 동화",
    "exposure": "노출 크기",
    "clutter": "배경 복잡도",
}

SCHEMA_VERSION = "0.1"
try:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.score.schema import ABILITY_LABELS as _ABILITY_LABELS  # type: ignore
    from src.score.schema import SCHEMA_VERSION as _SCHEMA_VERSION  # type: ignore

    ABILITY_LABELS: dict[str, str] = dict(_ABILITY_LABELS)
    SCHEMA_VERSION = _SCHEMA_VERSION
except Exception:  # 팀 스키마가 아직 없거나 경로가 다를 때
    ABILITY_LABELS = dict(_FALLBACK_LABELS)

DEFAULT_CONF_THRESHOLD = 0.25

# v0 노출 크기 기준: 사진 면적의 이 비율을 차지하면 0점
EXPOSURE_MAX_RATIO = 0.20

# (하한 점수, 칭호) — 높은 점수부터 검사
TITLES: list[tuple[int, str]] = [
    (90, "전설의 그림자"),
    (78, "완벽한 위장술사"),
    (64, "노련한 은둔자"),
    (50, "평범한 위장"),
    (34, "살짝 티나는 숨기"),
    (0, "AI에게 들킨 초보"),
]

_COMMENT_BANDS: dict[str, list[tuple[int, str]]] = {
    "evasion": [
        (85, "AI가 거의 확신하지 못했어요."),
        (70, "AI가 한참 헷갈려 했어요."),
        (50, "AI가 어렵지 않게 찾아냈어요."),
        (0, "AI가 바로 찾았어요."),
    ],
    "color_blend": [
        (80, "주변 색과 거의 완전히 어울려요."),
        (60, "주변 색과 잘 어울려요."),
        (40, "색 대비가 조금 눈에 띄어요."),
        (0, "색이 주변과 확실히 달라요."),
    ],
    "texture_blend": [
        (80, "질감이 배경과 잘 섞였어요."),
        (60, "무늬가 조금 튀어요."),
        (40, "윤곽이나 무늬가 드러나요."),
        (0, "질감 차이가 뚜렷해요."),
    ],
    "exposure": [
        (80, "아주 작게 숨어서 찾기 어려워요."),
        (60, "크기가 적당히 작아요."),
        (40, "몸이 꽤 크게 보여요."),
        (0, "너무 크게 드러나 있어요."),
    ],
    "clutter": [
        (80, "복잡한 곳을 잘 골랐어요."),
        (60, "배경에 숨을 요소가 어느 정도 있어요."),
        (40, "배경이 단조로워 눈에 띄기 쉬워요."),
        (0, "배경이 너무 비어 있어요."),
    ],
}


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _band(axis: str, score: Optional[int]) -> str:
    if score is None:
        return ""
    for threshold, text in _COMMENT_BANDS.get(axis, []):
        if score >= threshold:
            return text
    return ""


def _clamp_box(bbox: Any, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = (float(v) for v in bbox)
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = max(0.0, min(x1, float(width)))
    x2 = max(0.0, min(x2, float(width)))
    y1 = max(0.0, min(y1, float(height)))
    y2 = max(0.0, min(y2, float(height)))
    if x2 - x1 < 3:
        cx = (x1 + x2) / 2
        x1, x2 = max(0.0, cx - 1.5), min(float(width), cx + 1.5)
    if y2 - y1 < 3:
        cy = (y1 + y2) / 2
        y1, y2 = max(0.0, cy - 1.5), min(float(height), cy + 1.5)
    return [x1, y1, x2, y2]


def _expand_box(box: list[float], ratio: float, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    pad_x = (x2 - x1) * ratio
    pad_y = (y2 - y1) * ratio
    return [
        max(0.0, x1 - pad_x),
        max(0.0, y1 - pad_y),
        min(float(width), x2 + pad_x),
        min(float(height), y2 + pad_y),
    ]


def _small(crop: "Image.Image", limit: int = 256) -> "Image.Image":
    """통계 계산 속도를 위해 긴 변을 limit 이하로 줄입니다."""
    if max(crop.size) <= limit:
        return crop
    resized = crop.copy()
    resized.thumbnail((limit, limit))
    return resized


class ScoreConnector:
    """탐지 결과 + 이미지 픽셀 통계로 Score response 스키마를 만드는 어댑터."""

    def get_status(self) -> dict[str, Any]:
        return {
            "module": "score",
            "ready": True,
            "schema_version": SCHEMA_VERSION,
            "aggregate": "mean(v0)",
            "axes": [
                {"key": key, "label": label} for key, label in ABILITY_LABELS.items()
            ],
        }

    # ── 채점 ───────────────────────────────────────────────
    def evaluate(
        self,
        detection: Optional[dict[str, Any]],
        elapsed_sec: Optional[float] = None,
        image: Optional["Image.Image"] = None,
    ) -> dict[str, Any]:
        detection = detection or {}
        image_info = detection.get("image") or {}
        detections = detection.get("detections") or []

        width = int(image_info.get("width") or (image.width if image is not None else 0) or 0)
        height = int(image_info.get("height") or (image.height if image is not None else 0) or 0)
        meta = self._meta(detection)

        if not detections:
            return self._not_found(width, height, meta, elapsed_sec)

        best = max(detections, key=lambda d: float(d.get("confidence") or 0.0))
        confidence = _clamp(float(best.get("confidence") or 0.0), 0.0, 1.0)
        bbox = list(best.get("bbox") or [])

        scores: dict[str, Optional[int]] = {key: None for key in ABILITY_LABELS}
        scores["evasion"] = int(round((1.0 - confidence) * 100.0))

        if width > 0 and height > 0 and len(bbox) == 4:
            scores["exposure"] = self._exposure_score(bbox, width, height)
            scores.update(self._blend_scores(image, bbox, width, height))

        total = self._mean_total(scores)
        title = self._title(total)

        return {
            "schema_version": SCHEMA_VERSION,
            "found": True,
            "image": {"width": max(1, width), "height": max(1, height)},
            "detection": {"bbox": bbox, "confidence": round(confidence, 4)},
            "total_score": total,
            "title": title,
            "abilities": self._abilities(scores),
            "elapsed_sec": elapsed_sec,
            "meta": meta,
        }

    # ── 내부 계산 ──────────────────────────────────────────
    @staticmethod
    def _meta(detection: dict[str, Any]) -> dict[str, Any]:
        threshold = detection.get("conf_threshold")
        return {
            "model": detection.get("model"),
            "inference_ms": detection.get("inference_ms"),
            "conf_threshold": float(threshold)
            if isinstance(threshold, (int, float))
            else DEFAULT_CONF_THRESHOLD,
        }

    def _not_found(
        self,
        width: int,
        height: int,
        meta: dict[str, Any],
        elapsed_sec: Optional[float],
    ) -> dict[str, Any]:
        """AI가 못 찾음 → 판정 불가. 점수는 null, 화면 표현은 프론트가 정함."""
        return {
            "schema_version": SCHEMA_VERSION,
            "found": False,
            "image": {"width": max(1, width), "height": max(1, height)},
            "detection": None,
            "total_score": None,
            "title": None,
            "abilities": self._abilities({key: None for key in ABILITY_LABELS}),
            "elapsed_sec": elapsed_sec,
            "meta": meta,
        }

    @staticmethod
    def _abilities(scores: dict[str, Optional[int]]) -> list[dict[str, Any]]:
        abilities: list[dict[str, Any]] = []
        for key, label in ABILITY_LABELS.items():
            score = scores.get(key)
            score = int(score) if score is not None else None
            abilities.append(
                {
                    "key": key,
                    "label": label,
                    "score": score,
                    "comment": _band(key, score),
                }
            )
        return abilities

    @staticmethod
    def _mean_total(scores: dict[str, Optional[int]]) -> int:
        values = [float(v) for v in scores.values() if v is not None]
        if not values:
            return 0
        return int(round(_clamp(sum(values) / len(values))))

    @staticmethod
    def _title(total: int) -> str:
        return next((name for threshold, name in TITLES if total >= threshold), TITLES[-1][1])

    @staticmethod
    def _exposure_score(bbox: list[float], width: int, height: int) -> int:
        box = _clamp_box(bbox, width, height)
        area_ratio = (box[2] - box[0]) * (box[3] - box[1]) / float(width * height)
        return int(round(100.0 * (1.0 - min(1.0, area_ratio / EXPOSURE_MAX_RATIO))))

    def _blend_scores(
        self,
        image: Optional["Image.Image"],
        bbox: list[float],
        width: int,
        height: int,
    ) -> dict[str, Optional[int]]:
        """박스 안 vs 주변 띠의 색·질감·복잡도 비교. 불가능하면 null."""
        empty: dict[str, Optional[int]] = {
            "color_blend": None,
            "texture_blend": None,
            "clutter": None,
        }
        if image is None or ImageStat is None or width <= 0 or height <= 0:
            return empty

        box = _clamp_box(bbox, width, height)
        outer = _expand_box(box, 0.35, width, height)

        inner_area = (box[2] - box[0]) * (box[3] - box[1])
        outer_area = (outer[2] - outer[0]) * (outer[3] - outer[1])
        ring_area = outer_area - inner_area
        if ring_area <= 1:
            return empty  # 사진 전체를 덮는 박스 → 주변과 비교 불가

        try:
            inner_crop = _small(image.crop(tuple(int(round(v)) for v in box)).convert("RGB"))
            outer_crop = _small(image.crop(tuple(int(round(v)) for v in outer)).convert("RGB"))
            inner_rgb = ImageStat.Stat(inner_crop).mean[:3]
            outer_rgb = ImageStat.Stat(outer_crop).mean[:3]
            inner_edge = self._edge_mean(inner_crop)
            outer_edge = self._edge_mean(outer_crop)
        except Exception:
            return empty

        ring_rgb = [
            max(0.0, (outer_rgb[i] * outer_area - inner_rgb[i] * inner_area) / ring_area)
            for i in range(3)
        ]
        ring_edge = max(
            0.0, (outer_edge * outer_area - inner_edge * inner_area) / ring_area
        )

        color_dist = math.sqrt(sum((inner_rgb[i] - ring_rgb[i]) ** 2 for i in range(3)))
        texture_diff = abs(inner_edge - ring_edge)

        return {
            "color_blend": int(round(100.0 * (1.0 - min(1.0, color_dist / 150.0)))),
            "texture_blend": int(round(100.0 * (1.0 - min(1.0, texture_diff / 50.0)))),
            "clutter": int(round(100.0 * min(1.0, ring_edge / 40.0))),
        }

    @staticmethod
    def _edge_mean(crop: "Image.Image") -> float:
        if ImageFilter is None:
            return 0.0
        gray = crop.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        return float(ImageStat.Stat(edges).mean[0])


# app.py 가 `from score_connector import score_connector` 로 가져다 씁니다.
score_connector = ScoreConnector()
