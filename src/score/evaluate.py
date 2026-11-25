"""
meccha finder - 은폐 점수 계산 (v0)

DetectionResult + 원본 사진 → ScoreResponse (schemas/score_response.schema.json)
능력치의 의도와 근거는 docs/score-schema.md 4장 참고.

v0 계산식 요약 (모두 0~100, 높을수록 잘 숨음):
    evasion        (1 - confidence) × 100
    color_blend    박스 안 vs 주변 띠의 Lab 색 히스토그램 유사도 (1 - 바타차리야 거리)
    texture_blend  박스 안 vs 주변 띠의 기울기(크기·방향) 히스토그램 유사도
    exposure       사진 대비 박스 면적이 작을수록 높게: 1 - sqrt(면적비 / EXPOSURE_REF)
    clutter        주변 띠의 엣지 밀도가 높을수록 높게: 밀도 / CLUTTER_REF
    total          null 이 아닌 항목의 평균 (가중치는 3주차에 결정)
아래 상수는 실제 데이터로 맞춰 갈 값입니다.
"""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

from src.score.schema import ABILITY_LABELS, Ability, DetectionResult, ScoreResponse

RING_RATIO = 0.5        # 주변 띠 두께 = 박스 긴 변 × 이 비율
EXPOSURE_REF = 0.25     # 박스가 사진의 25% 이상이면 노출 크기 0점
CLUTTER_REF = 0.15      # 주변 엣지 밀도 15% 이상이면 배경 복잡도 100점
MIN_PIXELS = 30         # 박스나 띠가 이보다 작으면 계산하지 않음 (null)

TITLES = [  # (최소 점수, 칭호) - 임시. 화면 문구는 프론트가 바꿔도 됨
    (85, "전설의 카멜레온"),
    (70, "숙련된 은신술사"),
    (50, "아쉬운 견습 카멜레온"),
    (30, "눈에 띄는 초보"),
    (0, "숨바꼭질 입문자"),
]

COMMENTS = {  # key: (높음 ≥70, 중간 40~69, 낮음 <40)
    "evasion": ("AI가 한참 헷갈렸어요.", "AI가 조금 망설였어요.", "AI가 바로 찾았어요."),
    "color_blend": ("주변 색에 녹아들었어요.", "색이 조금 튀어요.", "색이 튀어요. 몸 색과 비슷한 곳을 골라 보세요."),
    "texture_blend": ("무늬까지 주변과 비슷해요.", "무늬가 조금 달라요.", "무늬가 튀어요. 질감이 비슷한 곳을 골라 보세요."),
    "exposure": ("몸이 조금만 보여요.", "몸이 꽤 보여요.", "몸이 크게 드러났어요. 물체 뒤에 숨거나 멀리 가 보세요."),
    "clutter": ("복잡한 곳이라 눈이 헷갈려요.", "주변이 적당히 복잡해요.", "주변이 단순해서 눈에 잘 띄어요."),
}


def _clip100(x: float) -> int:
    return int(round(min(100.0, max(0.0, x))))


def _masks(h: int, w: int, bbox: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """박스 안 마스크, 박스를 둘러싼 띠(링) 마스크"""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    inner = np.zeros((h, w), np.uint8)
    inner[y1:y2, x1:x2] = 1
    m = int(round(max(x2 - x1, y2 - y1) * RING_RATIO))
    outer = np.zeros((h, w), np.uint8)
    outer[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)] = 1
    ring = outer & (1 - inner)
    return inner.astype(bool), ring.astype(bool)


def _hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """정규화된 두 히스토그램의 유사도 0~1 (1 - 바타차리야 거리)"""
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a /= a.sum() or 1
    b /= b.sum() or 1
    return 1.0 - float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))


def _color_blend(lab: np.ndarray, inner: np.ndarray, ring: np.ndarray) -> float:
    bins, rng = [8, 8, 8], [0, 256, 0, 256, 0, 256]
    hi = cv2.calcHist([lab], [0, 1, 2], inner.astype(np.uint8), bins, rng)
    hr = cv2.calcHist([lab], [0, 1, 2], ring.astype(np.uint8), bins, rng)
    return _hist_similarity(hi, hr) * 100


def _texture_blend(mag: np.ndarray, ang: np.ndarray, inner: np.ndarray, ring: np.ndarray) -> float:
    def feat(mask: np.ndarray) -> np.ndarray:
        m, a = mag[mask], ang[mask]
        h_mag, _ = np.histogram(np.log1p(m), bins=16, range=(0, math.log1p(1500)))
        h_ang, _ = np.histogram(a % 180, bins=8, range=(0, 180), weights=m)
        h_mag = h_mag / max(h_mag.sum(), 1)
        h_ang = h_ang / max(h_ang.sum(), 1e-9)
        return np.concatenate([h_mag, h_ang])
    return _hist_similarity(feat(inner), feat(ring)) * 100


def _edge_density(gray: np.ndarray, mask: np.ndarray) -> float:
    med = float(np.median(gray))
    edges = cv2.Canny(gray, int(max(0, 0.66 * med)), int(min(255, 1.33 * med)) or 50)
    return float(edges[mask].mean() / 255.0)


def _comment(key: str, score: Optional[int]) -> str:
    if score is None:
        return ""
    high, mid, low = COMMENTS[key]
    return high if score >= 70 else mid if score >= 40 else low


def _title(total: int) -> str:
    return next(t for th, t in TITLES if total >= th)


def evaluate(det: DetectionResult, image_rgb: np.ndarray, elapsed_sec: Optional[float] = None) -> ScoreResponse:
    """det: 탐지 결과, image_rgb: 원본 사진 (H, W, 3) uint8 RGB"""
    meta = {"model": det.model, "inference_ms": det.inference_ms, "conf_threshold": det.conf_threshold}
    best = det.best
    if best is None:
        resp = ScoreResponse.not_found(det.width, det.height, meta)
        resp.elapsed_sec = elapsed_sec
        return resp

    h, w = image_rgb.shape[:2]
    inner, ring = _masks(h, w, best.bbox)
    scores: dict[str, Optional[int]] = {k: None for k in ABILITY_LABELS}

    scores["evasion"] = _clip100((1 - best.confidence) * 100)
    box_area = inner.sum()
    if box_area > 0:
        scores["exposure"] = _clip100((1 - math.sqrt(box_area / (h * w) / EXPOSURE_REF)) * 100)

    if box_area >= MIN_PIXELS and ring.sum() >= MIN_PIXELS:
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        scores["color_blend"] = _clip100(_color_blend(lab, inner, ring))
        scores["texture_blend"] = _clip100(_texture_blend(mag, ang, inner, ring))
        scores["clutter"] = _clip100(_edge_density(gray, ring) / CLUTTER_REF * 100)

    valid = [s for s in scores.values() if s is not None]
    total = int(round(sum(valid) / len(valid)))
    abilities = [Ability(k, scores[k], _comment(k, scores[k])) for k in ABILITY_LABELS]
    return ScoreResponse(
        width=det.width,
        height=det.height,
        detection=best,
        total_score=total,
        title=_title(total),
        abilities=abilities,
        elapsed_sec=elapsed_sec,
        meta=meta,
    )
