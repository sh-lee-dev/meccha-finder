# API 서버 실행 가이드

사진을 받아 YOLO로 캐릭터를 찾고, 은폐 점수를 계산해 JSON으로 돌려주는 서버예요. 응답 형식은 [score-schema.md](score-schema.md)를 따릅니다.

```
프론트(index.html) ──POST /detect──> src/api/main.py
                                        ├─ src/model/detect.py    YOLO 탐지 → Detection
                                        └─ src/score/evaluate.py  점수 계산 → Score response
                     <──── JSON ────────┘
```

## 1. 실행하기 (Windows 기준, 레포 루트에서)

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m uvicorn src.api.main:app --reload --port 8000
```

- 화면: http://127.0.0.1:8000 (서버가 `frontend/index.html`을 같이 보여줘요)
- API 문서(직접 사진 올려 보기): http://127.0.0.1:8000/docs
- 서버 상태: http://127.0.0.1:8000/api/status

처음 실행하면 기본 모델(`yolo26n.pt`)을 자동으로 내려받아요. 서버를 켤 때 모델을 미리 한 번 돌려 두기 때문에 시작에 몇 초 걸려요.

## 2. 설정 (환경변수)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MECCHA_WEIGHTS` | `yolo26n.pt` | 가중치 파일 경로. 학습한 `best.pt`가 생기면 이것만 바꾸면 돼요 |
| `MECCHA_CONF` | `0.25` | 이 신뢰도 이상이면 "찾았다" |
| `MECCHA_IMGSZ` | `640` | 추론 해상도. 캐릭터가 작으면 `1280` |
| `MECCHA_ANY_CLASS` | 꺼짐 | `1`이면 모든 클래스를 캐릭터로 취급 (기본 모델로 흐름만 테스트할 때) |
| `MECCHA_CORS_ORIGINS` | `*` | 허용할 프론트 주소. 배포 때는 GitHub Pages 주소로 좁히기 |

Windows PowerShell에서 설정하는 예:

```powershell
$env:MECCHA_WEIGHTS = "C:\Users\PC\Desktop\best.pt"
python -m uvicorn src.api.main:app --port 8000
```

**어떤 클래스를 캐릭터로 볼지**
- 클래스가 1개인 모델(우리가 학습한 `best.pt`)이면 그 클래스를 캐릭터로 봐요. 이름이 `chameleon`이 아니어도 돼요.
- 기본 모델(`yolo26n.pt`, 사람·버스 등 80종)에는 캐릭터 클래스가 없어서 항상 "못 찾음"이 나와요. 화면 연결만 확인하고 싶으면 `MECCHA_ANY_CLASS=1`로 켜세요. 그러면 사람이나 물체를 캐릭터처럼 다뤄요.

## 3. 주소

| 메서드 | 주소 | 설명 |
|---|---|---|
| `POST` | `/detect` | 사진 분석. 요청·응답 형식은 [score-schema.md](score-schema.md) 2~3장 |
| `GET` | `/api/status` | 서버·모델 상태 (가중치, 기준값, 클래스) |
| `GET` | `/` | 프론트 화면 |
| `GET` | `/docs` | 자동 생성 API 문서 |

`/api/detect`도 `/detect`와 똑같이 동작해요. 예전 `frontend/app.py` 주소를 쓰던 코드를 위한 호환용이에요.

## 4. 테스트

```bash
python -m pytest -q
```

모델 없이 가짜 탐지기로 돌려서, 응답이 스키마를 지키는지 확인해요. 확인하는 경우는 찾음, 못 찾음, 시간 값 전달, 에러 5종(사진 아님, 깨진 파일, 사진 없음, 잘못된 값, 10MB 초과), 모델 오류예요.

## 5. `frontend/app.py`와의 관계

`src/api/main.py`는 승우가 만든 `frontend/app.py`의 구조를 바탕으로 만들었어요. CORS, 상태 확인, 화면 제공은 그대로 가져왔고, 아래만 바꿨어요.

| 바뀐 점 | 이유 |
|---|---|
| 응답을 스키마(v0.1) 형식으로 | 프론트·점수·모델이 같은 약속을 쓰기 위해 |
| `yolo_connector`, `score_connector` → `src/model/detect.py`, `src/score/evaluate.py` | 커넥터 파일이 레포에 없어서, 각 담당 폴더에 실제 코드로 둠 |
| `/api/config`(실행 중 설정 변경) 제거 | 누구나 서버 설정을 바꿀 수 있어서 위험함. 환경변수로 대체 |
| 에러를 `error_response` 형식과 HTTP 코드로 | 프론트가 에러 문장을 그대로 보여줄 수 있게 |

`frontend/app.py`를 지울지는 승우와 정해요.

## 6. 점수 계산 v0의 한계

- 계산 상수(`EXPOSURE_REF`, `CLUTTER_REF`, `RING_RATIO`)는 임시값이에요. 실제 캡처 사진으로 맞춰야 해요 (`src/score/evaluate.py` 위쪽).
- 총점은 다섯 항목의 단순 평균이에요. 가중치는 3주차에 정해요.
- 색·질감은 사각형 박스 안 전체를 보기 때문에, 박스 안의 배경도 함께 섞여 계산돼요.
