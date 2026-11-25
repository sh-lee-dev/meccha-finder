"""scripts/evaluate_folder.py 의 계산 함수 테스트 (모델 없이 실행)"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_folder as ev  # noqa: E402


def test_label_key_strips_roboflow_suffix():
    assert ev.label_key("frame_001_png.rf.ab12cd") == "frame_001"
    assert ev.label_key("jungle_frame_sh003_jpg.rf.99") == "jungle_frame_sh003"
    assert ev.label_key("frame_002") == "frame_002"


def test_iou():
    assert ev.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert ev.iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(ev.iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9


def test_match_counts():
    gts = [[0, 0, 10, 10], [50, 50, 60, 60]]
    preds = [[0, 0, 10, 10], [100, 100, 110, 110]]  # 하나 맞힘, 하나 헛짚음
    tp, fp, fn, ious = ev.match(preds, gts)
    assert (tp, fp, fn) == (1, 1, 1) and ious == [1.0]


def test_read_gt_boxes(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
    assert ev.read_gt_boxes(p, 100, 50) == [[40.0, 15.0, 60.0, 35.0]]


def test_spearman():
    assert ev.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert ev.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    assert ev.spearman([1, 2], [1, 2]) is None
