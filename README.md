# 메챠 파인더 (Meccha Finder)

> 숨은 캐릭터가 **얼마나 잘 숨었는지**를 YOLO가 찾아내고, 점수로 평가해 주는 위장 탐지(camouflage detection) 미니게임

**상태:** 1차 MVP 준비 중

---

## 1. 프로젝트 소개

숨은 캐릭터 찾기 게임에서 영감을 받은 팀 프로젝트입니다.
사람이 직접 만든 데이터로 YOLO 객체 탐지(object detection) 모델을 학습시키고, 이미지 속 캐릭터를 모델이 얼마나 쉽게 찾아내는지를 기준으로 **은폐 점수**를 계산합니다.

- 모델이 쉽게 찾는다 → 덜 숨은 것
- 모델이 못 찾거나 확신이 낮다 → 잘 숨은 것

## 2. 주요 기능

| 구분 | 기능 | 단계 |
|---|---|---|
| 탐지 | 이미지에서 캐릭터 위치(박스)와 신뢰도(confidence) 추출 | MVP |
| 평가 | 탐지 신뢰도 + 찾는 데 걸린 시간 기반 점수 계산 | MVP |
| 결과 | 총점, 평가 기준별 코멘트, 오각형/육각형 능력치 차트 | 3~4주차 |
| 시각화 | 캐릭터 윤곽선 표시 | 확장 (세그멘테이션 전환 필요) |

## 3. 동작 흐름

```mermaid
flowchart LR
    A[이미지 입력] --> B[YOLO 탐지<br/>박스 + 신뢰도]
    B --> C[점수 계산<br/>신뢰도 + 소요 시간]
    C --> D[결과 화면<br/>총점 / 코멘트 / 능력치]
```

각 단계는 팀원별 모듈로 나뉘어 있고, 모듈 사이의 입출력 형식은 [팀 지침서](docs/team-guide.md#6-모듈-간-입출력-약속)에 정의합니다.

## 4. 점수 평가 방식 (초안)

- 입력: 탐지 신뢰도, 발견 소요 시간 (측정 방식은 [요구사항](docs/requirements.md#7-미결정-사항)에서 확정 예정)
- 직관: 신뢰도가 낮을수록, 찾는 데 오래 걸릴수록 높은 점수
- 출력: 0~100 총점, 기준별 능력치, 기준별 코멘트
- 세부 공식과 능력치 축은 3주차에 팀 합의 후 확정합니다.

## 5. 기술 스택

| 영역 | 사용 기술 |
|---|---|
| 언어 | Python |
| 객체 탐지 | Ultralytics YOLO (시작 모델: `yolo11n`) |
| 영상/이미지 처리 | OpenCV |
| 라벨링 | Roboflow 등 (YOLO 형식으로 내보내기) |
| UI | 미정 |
| 협업 | GitHub (코드·핵심 문서), Google Drive (일정·진행·회의록·데이터·모델 가중치) |

## 6. 프로젝트 구조

```
meccha-finder/
├── README.md
├── requirements.txt          # 예정
├── docs/
│   ├── requirements.md       # 요구사항
│   ├── decisions.md          # 결정 기록
│   └── team-guide.md         # 팀 지침서
├── src/
│   ├── model/                # YOLO 학습·추론
│   ├── score/                # 점수 알고리즘
│   └── ui/                   # 화면·통합
├── scripts/
│   └── prepare_dataset.py    # 프레임 추출, train/val 분할 (추가 예정)
└── data/                     # Git 제외, Drive에서 내려받아 사용
```

## 7. 시작하기

```bash
git clone https://github.com/sh-lee-dev/meccha-finder.git
cd meccha-finder

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

실습실에서는 매번 환경 설정이 필요하므로 체크리스트를 [팀 지침서](docs/team-guide.md#3-실습실에서-시작할-때-체크리스트)에 따로 정리해 두었습니다.

학습과 예측 (데이터셋 준비 후):

```bash
yolo detect train data=data/meccha-dataset/data.yaml model=yolo11n.pt epochs=50 imgsz=640
yolo detect predict model=runs/detect/train/weights/best.pt source=<이미지 경로>
```

## 8. 데이터셋

- 출처: 유튜브 영상과 직접 플레이한 화면에서 프레임을 추출해 사용 (자동 라벨링 없이 직접 라벨링)
- 1차 범위: 클래스 1개(`chameleon`), 자세 1개 고정, 바운딩 박스(bounding box)만 사용
- 저장 위치: Google Drive 공유 폴더 (원본 영상과 프레임은 저장소에 올리지 않음)
- 만드는 방법: [팀 지침서](docs/team-guide.md#7-google-drive-사용법)와 `scripts/prepare_dataset.py` 참고

## 9. 로드맵

| 주차 | 기간 | 목표 |
|---|---|---|
| 1주차 | 11/11 ~ 11/15 | 환경·규칙 정리, 데이터 MVP, 모듈 간 입출력 합의 |
| 2주차 | 11/16 ~ 11/22 | 첫 학습, 모듈별 v0 (더미 데이터로 단독 동작) |
| 3주차 | 11/23 ~ 11/29 | 모듈 통합(end-to-end 데모), 데이터 확장 |
| 4주차 | 11/30 ~ 12/06 | 성능·점수 고도화, 능력치 차트, 포트폴리오 정리 |

작업 단위 일정(WBS)과 주간 진행상황은 팀 Google Drive에서 관리합니다.

## 10. 팀

| 이름 | 역할 | 브랜치 |
|---|---|---|
| 이상현 (조장) | 점수 알고리즘·백엔드 | `score/algorithm` |
| 이한울 | YOLO·모델 | `yolo/model` |
| 한승우 | UI·통합 | `ui/merge` |

## 11. 문서

- [요구사항](docs/requirements.md)
- [결정 기록](docs/decisions.md)
- [팀 지침서](docs/team-guide.md)
