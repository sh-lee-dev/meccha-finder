"""
meccha finder - 일괄 평가 스크립트

사진 폴더를 한 번에 돌려서 모델 성능과 점수 분포를 확인합니다.
서버(/detect)와 똑같은 detect() + evaluate()를 쓰므로 결과가 서버와 같아요.

사용법 (레포 루트에서):
    # 기본: 레포의 best.pt 로 폴더 평가
    python scripts/evaluate_folder.py --images 사진폴더

    # 라벨(YOLO .txt)이 있으면 정답과 비교해 정밀도·재현율까지
    python scripts/evaluate_folder.py --images 사진폴더 --labels 라벨폴더

    # 모델 두 개 비교 (v1 vs v2)
    python scripts/evaluate_folder.py --images 사진폴더 --labels 라벨폴더 --weights best.pt best_v2.pt

    # 사람이 매긴 "잘 숨은 순위"와 점수 순서 비교 (CSV: file,rank  / rank 1 = 가장 잘 숨음)
    python scripts/evaluate_folder.py --images 사진폴더 --ranking ranking.csv

결과 (--out 폴더, 기본 eval_results/):
    <모델이름>/results.csv      사진별 결과 (찾음 여부, 신뢰도, 박스, 총점, 능력치, 정답 IoU)
    <모델이름>/summary.json     요약 (찾음 비율, 점수 분포, 정밀도·재현율, 순위 상관)
    <모델이름>/boxes/*.jpg      박스 그린 이미지 (빨강: 찾음, 노랑: 기준 미만 후보, 초록: 정답)
    <모델이름>/sheet.jpg        한눈에 보기 (모아 찍은 이미지)
    compare.md                  모델 여러 개를 넣었을 때 비교표
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.detect import Detector, default_weights  # noqa: E402
from src.score.evaluate import evaluate  # noqa: E402
from src.score.schema import ABILITY_LABELS, DetectionResult  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
PEEK_CONF = 0.05      # 기준 미만이지만 모델이 의심한 후보도 이미지에 표시
IOU_MATCH = 0.5       # 정답과 이 이상 겹치면 맞게 찾은 것


# ---------- 라벨·지표 ----------
def label_key(stem: str) -> str:
    """Roboflow 가 붙이는 꼬리(_png.rf.xxxx)를 떼고 비교: frame_001_png.rf.ab12 -> frame_001"""
    return stem.split("_png.rf.")[0].split("_jpg.rf.")[0].split("_jpeg.rf.")[0]


def load_labels(label_dir: Optional[Path]) -> dict[str, Path]:
    if not label_dir:
        return {}
    return {label_key(p.stem): p for p in label_dir.rglob("*.txt")}


def read_gt_boxes(path: Path, w: int, h: int) -> list[list[float]]:
    """YOLO 라벨(class cx cy bw bh, 0~1) → 픽셀 [x1, y1, x2, y2]"""
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(preds: list[list[float]], gts: list[list[float]]) -> tuple[int, int, int, list[float]]:
    """탐욕 매칭 → (TP, FP, FN, 맞춘 박스들의 IoU)"""
    used, ious = set(), []
    for p in preds:  # preds 는 신뢰도 높은 순
        best_j, best_iou = -1, 0.0
        for j, g in enumerate(gts):
            if j not in used and (v := iou(p, g)) > best_iou:
                best_j, best_iou = j, v
        if best_j >= 0 and best_iou >= IOU_MATCH:
            used.add(best_j)
            ious.append(best_iou)
    tp = len(ious)
    return tp, len(preds) - tp, len(gts) - tp, ious


def spearman(x: list[float], y: list[float]) -> Optional[float]:
    """순위 상관 (동점은 평균 순위). 값이 2개 미만이면 None"""
    if len(x) < 3:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return round(num / den, 3) if den else None


def describe(values: list[float]) -> Optional[dict]:
    if not values:
        return None
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 1),
        "std": round(statistics.pstdev(values), 1),
        "min": min(values),
        "max": max(values),
    }


# ---------- 그림 ----------
def draw(img: Image.Image, peek: DetectionResult, conf: float, gts: list[list[float]], title: str) -> Image.Image:
    im = img.copy()
    d = ImageDraw.Draw(im)
    lw = max(2, im.width // 300)
    for g in gts:
        d.rectangle(g, outline=(0, 200, 0), width=lw)
    for det in peek.detections[:5]:
        color = (255, 0, 0) if det.confidence >= conf else (255, 210, 0)
        d.rectangle(det.bbox, outline=color, width=lw)
        d.text((det.bbox[0] + 3, det.bbox[1] + 3), f"{det.confidence:.2f}", fill=color)
    d.rectangle([0, 0, min(im.width, 12 + 7 * len(title)), 16], fill=(0, 0, 0))
    d.text((4, 2), title, fill=(255, 255, 255))
    return im


def contact_sheet(images: list[Image.Image], path: Path, cols: int = 5, h: int = 200) -> None:
    if not images:
        return
    tiles = [im.resize((max(1, int(im.width * h / im.height)), h)) for im in images]
    cw = max(t.width for t in tiles)
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * (cw + 6), rows * (h + 6)), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (cw + 6), (i // cols) * (h + 6)))
    sheet.save(path, quality=85)


# ---------- 실행 ----------
def run_one(weights: str, root: Path, images: list[Path], labels: dict[str, Path], ranking: dict[str, float],
            out: Path, conf: float, imgsz: int) -> dict:
    name = Path(weights).stem
    mdir = out / name
    (mdir / "boxes").mkdir(parents=True, exist_ok=True)
    det = Detector(weights=weights, conf=PEEK_CONF, imgsz=imgsz)
    det.load()
    if det.target_mode.startswith("no_target"):
        print(f"[경고] {weights}: 캐릭터 클래스가 없는 모델이라 항상 '못 찾음'이 나와요.")

    rows, drawn = [], []
    tp = fp = fn = 0
    all_ious: list[float] = []
    for i, p in enumerate(images, 1):
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        peek = det.detect(img)  # 낮은 기준으로 한 번만 추론
        hits = [d for d in peek.detections if d.confidence >= conf]
        result = evaluate(DetectionResult(img.width, img.height, hits, conf_threshold=conf,
                                          inference_ms=peek.inference_ms, model=peek.model), np.asarray(img)).to_dict()
        key = label_key(p.stem)
        gts = read_gt_boxes(labels[key], img.width, img.height) if key in labels else []
        row = {
            "file": str(p.relative_to(root)),
            "found": result["found"],
            "confidence": result["detection"]["confidence"] if result["detection"] else None,
            "bbox": result["detection"]["bbox"] if result["detection"] else None,
            "n_boxes": len(hits),
            "total_score": result["total_score"],
            **{a["key"]: a["score"] for a in result["abilities"]},
            "inference_ms": peek.inference_ms,
        }
        if labels:
            if key in labels:
                t, f_, n, ious = match([d.bbox for d in hits], gts)
                tp, fp, fn = tp + t, fp + f_, fn + n
                all_ious += ious
                row.update({"gt_boxes": len(gts), "tp": t, "fp": f_, "fn": n,
                            "best_iou": round(max((iou(d.bbox, g) for d in hits for g in gts), default=0.0), 3)})
            else:
                row.update({"gt_boxes": None, "tp": None, "fp": None, "fn": None, "best_iou": None})
        if ranking:
            row["human_rank"] = ranking.get(key)
        rows.append(row)
        im = draw(img, peek, conf, gts, f"{p.stem}  {result['total_score'] if result['found'] else '-'}")
        im.save(mdir / "boxes" / f"{p.stem}.jpg", quality=85)
        drawn.append(im)
        print(f"  [{name}] {i}/{len(images)} {p.name}: "
              f"{'찾음 %.2f' % row['confidence'] if row['found'] else '못 찾음'}  총점 {row['total_score']}")

    # CSV
    with open(mdir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:  # 엑셀 한글 깨짐 방지
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    found_rows = [r for r in rows if r["found"]]
    summary = {
        "weights": weights,
        "target_mode": det.target_mode,
        "conf_threshold": conf,
        "images": len(rows),
        "found": len(found_rows),
        "found_rate": round(len(found_rows) / len(rows), 3) if rows else None,
        "total_score": describe([r["total_score"] for r in found_rows]),
        "abilities": {k: describe([r[k] for r in found_rows if r[k] is not None]) for k in ABILITY_LABELS},
        "inference_ms": describe([r["inference_ms"] for r in rows if r["inference_ms"] is not None]),
    }
    if labels:
        summary["labels"] = {
            "images_with_label": sum(1 for r in rows if r.get("gt_boxes") is not None),
            "iou_match": IOU_MATCH,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "mean_iou_of_matches": round(statistics.mean(all_ious), 3) if all_ious else None,
        }
    if ranking:
        pairs = [(r["human_rank"], r["total_score"]) for r in rows
                 if r.get("human_rank") is not None and r["total_score"] is not None]
        # rank 1 = 가장 잘 숨음 → 점수가 높을수록 rank 가 작아야 하므로 부호를 뒤집어 비교
        rho = spearman([-a for a, _ in pairs], [b for _, b in pairs])
        summary["ranking"] = {"pairs": len(pairs), "spearman": rho,
                              "note": "1에 가까울수록 사람 순위와 점수 순서가 같음 (0.5 이상이면 쓸 만함)"}
    (mdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheet(drawn, mdir / "sheet.jpg")
    return summary


def write_compare(summaries: list[dict], path: Path) -> None:
    head = ["항목"] + [Path(s["weights"]).stem for s in summaries]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]

    def row(label, fn):
        lines.append("| " + " | ".join([label] + [fn(s) for s in summaries]) + " |")

    row("찾음 비율", lambda s: f"{s['found']}/{s['images']} ({s['found_rate']:.0%})" if s["found_rate"] is not None else "-")
    if all("labels" in s for s in summaries):
        row("정밀도 (IoU≥0.5)", lambda s: str(s["labels"]["precision"]))
        row("재현율 (IoU≥0.5)", lambda s: str(s["labels"]["recall"]))
        row("평균 IoU (맞춘 것)", lambda s: str(s["labels"]["mean_iou_of_matches"]))
    row("총점 평균 (찾은 사진)", lambda s: str(s["total_score"]["mean"]) if s["total_score"] else "-")
    row("총점 범위", lambda s: f"{s['total_score']['min']}~{s['total_score']['max']}" if s["total_score"] else "-")
    if all("ranking" in s for s in summaries):
        row("사람 순위 상관", lambda s: str(s["ranking"]["spearman"]))
    row("추론 시간 평균(ms)", lambda s: str(s["inference_ms"]["mean"]) if s["inference_ms"] else "-")
    path.write_text("# 모델 비교\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))


def load_ranking(path: Optional[Path]) -> dict[str, float]:
    if not path:
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {label_key(Path(r["file"]).stem): float(r["rank"]) for r in csv.DictReader(f) if r.get("rank")}


def main() -> None:
    ap = argparse.ArgumentParser(description="meccha finder 일괄 평가")
    ap.add_argument("--images", required=True, type=Path, help="사진 폴더 (하위 폴더 포함)")
    ap.add_argument("--labels", type=Path, help="YOLO 라벨(.txt) 폴더, 선택")
    ap.add_argument("--weights", nargs="+", help="가중치 파일 (여러 개면 비교). 기본: 레포의 best.pt")
    ap.add_argument("--ranking", type=Path, help="사람 순위 CSV (file,rank), 선택")
    ap.add_argument("--conf", type=float, default=0.25, help="찾음 기준 신뢰도 (서버와 같게, 기본 0.25)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, default=Path("eval_results"))
    args = ap.parse_args()

    images = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in IMG_EXTS)
    if not images:
        raise SystemExit(f"사진이 없어요: {args.images}")
    labels = load_labels(args.labels)
    ranking = load_ranking(args.ranking)
    weights_list = args.weights or [default_weights()]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"사진 {len(images)}장, 라벨 {len(labels)}개, 모델 {len(weights_list)}개 → {args.out}/")

    summaries = [run_one(w, args.images, images, labels, ranking, args.out, args.conf, args.imgsz) for w in weights_list]
    if len(summaries) > 1:
        write_compare(summaries, args.out / "compare.md")
    for s in summaries:
        lab = s.get("labels")
        print(f"\n[{Path(s['weights']).stem}] 찾음 {s['found']}/{s['images']}"
              + (f" | 정밀도 {lab['precision']} 재현율 {lab['recall']}" if lab else "")
              + (f" | 총점 평균 {s['total_score']['mean']}" if s["total_score"] else "")
              + (f" | 사람 순위 상관 {s['ranking']['spearman']}" if s.get("ranking") else ""))
    print(f"\n자세한 결과: {args.out.resolve()}")


if __name__ == "__main__":
    main()
