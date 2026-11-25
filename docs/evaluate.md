# 모델·점수 일괄 평가 가이드

`scripts/evaluate_folder.py`는 사진 폴더를 한 번에 돌려서 **모델이 얼마나 잘 찾는지**와 **점수가 어떻게 나오는지**를 보여줘요. 서버(`POST /detect`)와 똑같은 `detect()` + `evaluate()`를 써서, 결과가 서버 화면과 같아요.

## 언제 쓰나
| 상황 | 명령 |
|---|---|
| 새 모델이 나왔을 때 성능 확인 | `--images 확인용폴더 --labels 확인용라벨` |
| 모델 v1과 v2 비교 | `--weights best.pt best_v2.pt` |
| 점수 기준값 튜닝 전후 비교 | 같은 폴더로 튜닝 전후 실행 |
| 점수가 사람 판단과 맞는지 확인 | `--ranking ranking.csv` |

## 실행 (레포 루트에서)

```bash
# 기본: 레포의 best.pt 로 폴더 평가
python scripts/evaluate_folder.py --images 사진폴더

# 정답 라벨과 비교 (정밀도·재현율)
python scripts/evaluate_folder.py --images 사진폴더 --labels 라벨폴더

# v1 vs v2 비교
python scripts/evaluate_folder.py --images 사진폴더 --labels 라벨폴더 --weights best.pt best_v2.pt

# 사람이 매긴 순위와 비교
python scripts/evaluate_folder.py --images 사진폴더 --ranking ranking.csv
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--images` | (필수) | 사진 폴더. 하위 폴더까지 읽어요 |
| `--labels` | 없음 | YOLO 라벨(`.txt`) 폴더. Roboflow가 붙인 `_png.rf.xxxx` 꼬리가 있어도 사진과 짝지어져요 |
| `--weights` | 레포의 `best.pt` | 여러 개 넣으면 비교표(`compare.md`)를 만들어요 |
| `--ranking` | 없음 | 사람 순위 CSV |
| `--conf` | `0.25` | "찾았다" 기준. 서버와 같은 값 |
| `--out` | `eval_results/` | 결과 폴더 (`.gitignore`에 걸려 있어 커밋되지 않음) |

## 결과물

```
eval_results/
├── compare.md            모델 여러 개 비교표
└── best/                 모델 이름별 폴더
    ├── summary.json      요약
    ├── results.csv       사진별 결과 (엑셀로 열림)
    ├── sheet.jpg         박스 그린 사진 모음
    └── boxes/*.jpg       사진별 박스 이미지
```

**박스 색**: 빨강 = 찾음(기준 이상), 노랑 = 기준 미만이지만 모델이 의심한 후보, 초록 = 정답 라벨

**요약에 나오는 값**
| 값 | 뜻 |
|---|---|
| 찾음 비율 | 신뢰도 0.25 이상 박스가 나온 사진 비율 |
| 정밀도 | 찾았다고 한 박스 중 정답과 50% 이상 겹친 비율. 낮으면 엉뚱한 곳을 잡음 |
| 재현율 | 정답 중 찾아낸 비율. 낮으면 캐릭터를 놓침 |
| 평균 IoU | 맞춘 박스가 정답과 얼마나 겹치는지 (1이면 완전히 같음) |
| 총점·능력치 분포 | 찾은 사진의 평균, 표준편차, 최소, 최대 |
| 사람 순위 상관 | -1~1. 1에 가까울수록 사람이 매긴 순서와 점수 순서가 같음. 0.5 이상이면 쓸 만함 |

정밀도와 재현율은 **라벨 파일이 있는 사진만** 계산해요. 캐릭터가 없는 사진은 빈 `.txt`를 두면 "없음이 정답"으로 계산돼요.

## 사람 순위 CSV 만들기
셋이 같은 사진 10장 정도를 보고 **"가장 잘 숨은 사진 = 1"** 순서로 순위를 매겨요. 셋의 평균 순위를 쓰면 더 좋아요.

```csv
file,rank
frame_021.png,1
frame_025.png,2
frame_030.png,3
```

## 주의
- **학습에 쓴 사진으로 평가하면 실제보다 좋게 나와요.** 성능 확인용 사진은 학습에서 따로 빼 두세요.
- 기본 모델(`yolo26n.pt`)에는 캐릭터 클래스가 없어서 항상 "못 찾음"이 나와요. 경고 문구가 함께 떠요.
