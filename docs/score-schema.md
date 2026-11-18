# 점수 데이터 스키마 (v0.1 초안)

모델, 점수, 프론트가 주고받는 데이터의 모양입니다. 기준 파일은 `schemas/` 폴더의 JSON Schema이고, 예시 JSON은 `schemas/examples/`에 있어요. 프론트는 서버 없이도 예시 JSON으로 화면을 만들 수 있습니다.

```
사진 ──> [model] detect() ──(Detection)──> [score] evaluate() ──(Score response)──> [UI] 결과 화면
```

| 구간 | 스키마 파일 | 예시 | 보는 사람 |
|---|---|---|---|
| 모델 → 점수 | `schemas/detection.schema.json` | `detection_found.json`, `detection_not_found.json` | 한울, 상현 |
| 서버 → UI (성공) | `schemas/score_response.schema.json` | `response_found.json`, `response_not_found.json` | 상현, 승우 |
| 서버 → UI (실패) | `schemas/error_response.schema.json` | `response_error.json` | 상현, 승우 |

Python 코드에서는 `src/score/schema.py`의 클래스를 쓰면 같은 형식으로 만들어져요.

## 1. 모델 → 점수 (Detection)

```json
{
  "schema_version": "0.1",
  "image": { "width": 1390, "height": 781 },
  "detections": [
    { "bbox": [412.0, 168.0, 538.0, 305.0], "confidence": 0.87, "class_name": "chameleon" }
  ],
  "conf_threshold": 0.25,
  "inference_ms": 41.5,
  "model": "best_v1.pt"
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `image.width`, `image.height` | 정수 | **원본** 사진 크기(px) |
| `detections` | 배열 | 신뢰도 높은 순. 못 찾으면 `[]`. 다중 캐릭터 확장을 위해 처음부터 배열로 둠 |
| `detections[].bbox` | `[x1, y1, x2, y2]` | 원본 사진 기준 **픽셀** 좌표 (왼쪽 위, 오른쪽 아래) |
| `detections[].confidence` | 0~1 | 탐지 신뢰도 |
| `detections[].class_name` | `"chameleon"` | 클래스는 하나뿐 |
| `conf_threshold` | 0~1 | 이 값 이상인 박스만 `detections`에 넣음. **찾았다/못 찾았다의 기준** |
| `inference_ms`, `model` | 선택 | 기록용. 점수 계산에는 쓰지 않음 |

YOLO 결과에서 `results[0].boxes.xyxy`, `.conf`를 그대로 옮기면 돼요. YOLO는 학습·추론 때 내부에서 리사이즈해도 원본 기준 좌표를 돌려줍니다.

## 2. 프론트 → 서버 (요청)

`POST /detect`, `multipart/form-data` (지금 프론트 방식 그대로)

| 필드 | 타입 | 필수 | 규칙 |
|---|---|---|---|
| `image` | 파일 | 필수 | JPG/PNG/WEBP, 10MB 이하 |
| `elapsed_sec` | 숫자(초) | 선택 | 사람이 찾은 시간. **지금은 보내지 않음** (나중에 쓸 자리만 확보) |

## 3. 서버 → UI (응답)

```json
{
  "schema_version": "0.1",
  "found": true,
  "image": { "width": 1390, "height": 781 },
  "detection": { "bbox": [412.0, 168.0, 538.0, 305.0], "confidence": 0.87 },
  "total_score": 50,
  "title": "아쉬운 견습 카멜레온",
  "abilities": [
    { "key": "evasion",       "label": "탐지 회피",   "score": 13, "comment": "AI가 바로 찾았어요." },
    { "key": "color_blend",   "label": "색 동화",     "score": 72, "comment": "주변 색과 잘 어울려요." },
    { "key": "texture_blend", "label": "질감 동화",   "score": 58, "comment": "무늬가 조금 튀어요." },
    { "key": "exposure",      "label": "노출 크기",   "score": 40, "comment": "몸이 꽤 크게 보여요." },
    { "key": "clutter",       "label": "배경 복잡도", "score": 66, "comment": "복잡한 곳을 잘 골랐어요." }
  ],
  "elapsed_sec": null,
  "meta": { "model": "best_v1.pt", "inference_ms": 41.5, "conf_threshold": 0.25 }
}
```

| 필드 | 타입 | 설명 |
|---|---|---|
| `found` | bool | AI가 캐릭터를 찾았는지 |
| `image` | 객체 | 원본 사진 크기. 박스를 % 위치로 그릴 때 사용 |
| `detection` | 객체 또는 `null` | 화면에 그릴 대표 박스 1개 (신뢰도 최고) |
| `total_score` | 정수 0~100 또는 `null` | 은폐 총점. **높을수록 잘 숨음** |
| `title` | 문자열 또는 `null` | 점수 구간별 칭호 (임시) |
| `abilities` | 배열 | 능력치 차트용. **배열 순서 = 차트 순서**, 개수는 고정하지 않음 |
| `abilities[].key` | 영문 키 | **고정 이름.** 프론트는 이 키로 화면 이름과 아이콘을 정함 |
| `abilities[].label` | 문자열 | 임시 이름. 프론트가 바꿔 써도 됨 |
| `abilities[].score` | 정수 0~100 또는 `null` | 계산 못 한 항목은 `null` |
| `abilities[].comment` | 문자열 | 항목별 한 줄 코멘트 (없으면 `""`) |
| `elapsed_sec` | `null` | 지금은 항상 `null` |
| `meta` | 객체 | 기록용 (모델, 추론 시간, 신뢰도 기준값). 화면에는 쓰지 않음 |

### 못 찾았을 때 (`found: false`)
서버는 **판정 불가**라는 사실만 보내고, 화면 표현은 프론트가 정합니다.

| 필드 | 값 |
|---|---|
| `detection` | `null` |
| `total_score` | `null` |
| `title` | `null` |
| `abilities[].score` | 모두 `null` |

AI가 못 찾은 이유는 서버도 알 수 없어요. 정말 잘 숨었을 수도, 사진에 캐릭터가 없을 수도, 모델이 약할 수도 있어요. 그래서 100점으로 위장하지 않고 `null`로 보냅니다. 예시는 `schemas/examples/response_not_found.json`에 있어요.

### 실패했을 때
HTTP 4xx/5xx와 함께 보냅니다. `message`는 화면에 그대로 보여줘도 되는 문장이에요.

```json
{ "error": { "code": "INVALID_IMAGE", "message": "사진을 읽을 수 없어요. JPG, PNG, WEBP 파일로 다시 올려 주세요." } }
```

| code | HTTP | 언제 |
|---|---|---|
| `INVALID_IMAGE` | 400 | `image`가 없거나 사진이 아님 |
| `IMAGE_TOO_LARGE` | 413 | 10MB 초과 |
| `INVALID_PARAM` | 400 | `elapsed_sec` 등 다른 값이 잘못됨 |
| `MODEL_ERROR` | 500 | 모델 추론 실패 |
| `INTERNAL_ERROR` | 500 | 그 밖의 서버 오류 |

## 4. 능력치 항목 (5각형, v0)

YOLO는 캐릭터 윤곽이 아니라 **사각형 박스**만 줘요. 그래서 박스, 원본 사진, 신뢰도만으로 계산할 수 있는 항목만 골랐어요. 계산식은 v0 제안이고 3주차에 확정합니다.

| key | 임시 이름 | 계산 방법 (v0) | 근거 |
|---|---|---|---|
| `evasion` | 탐지 회피 | `(1 - confidence) × 100` | AI 눈에 얼마나 안 보이는지를 직접 잼 |
| `color_blend` | 색 동화 | 박스 안과 박스 주변 띠의 색 분포가 비슷할수록 높게 | 위장 평가에서 널리 쓰는 색 유사도 |
| `texture_blend` | 질감 동화 | 박스 안과 주변 띠의 엣지·무늬 분포가 비슷할수록 높게 | 무늬가 주변과 같으면 눈에 덜 띔 |
| `exposure` | 노출 크기 | 사진 대비 박스 면적이 작을수록 높게 | 보이는 면적이 작을수록 찾기 어려움 |
| `clutter` | 배경 복잡도 | 박스 주변 띠의 엣지 밀도가 높을수록 높게 | 주변이 복잡할수록 찾기 어려움 |

- **총점**: v0에서는 `null`이 아닌 항목의 평균(반올림)이에요. 예시의 50점은 13, 72, 58, 40, 66의 평균이에요. 가중치는 3주차에 정해요.
- **6각형 확장 후보**: `find_difficulty` (해상도별 발견 난이도). 같은 사진을 낮은 해상도부터 키워 가며 몇 단계에서 찾는지 봐요. `exposure`와 겹치는지 데이터로 확인한 뒤 추가하고, 추가해도 배열에 한 항목만 늘어나서 형식은 그대로예요.
- **쓰지 않기로 한 것**
  - 모델 추론 시간: 숨은 정도와 관계없이 거의 일정하고, 서버 상태에 따라 흔들려요. `meta`에 기록만 해요.
  - 가림 정도, 경계 융합: 캐릭터 윤곽(마스크)이 필요해요. 세그멘테이션으로 확장하면 다시 검토해요.
  - 위치(중앙에서의 거리): 근거가 약해서 `clutter`로 대체했어요.

**타당성 확인 방법**: 라벨링한 사진 10장을 셋이 각자 "잘 숨은 순서"로 줄 세우고, 점수 순서와 비교해요(순위 상관). 사람 판단과 엇나가는 항목은 빼거나 가중치를 낮춰요.

## 5. 프론트에서 바꿀 점

지금 `frontend/index.html`의 `mockResult()` 형식과 비교하면 이 부분이 바뀌어요.

| 지금 | 바뀐 형식 | 이유 |
|---|---|---|
| `d.bbox`, `d.confidence` | `d.detection.bbox`, `d.detection.confidence` | 못 찾았을 때 `detection: null`로 한 번에 구분 |
| `abilities: {'가림 정도': 75, ...}` | `abilities: [{key, label, score, comment}, ...]` | 순서 보장, 이름은 프론트가 `key`로 자유롭게 정함 |
| `comments: {...}` (따로) | 각 `abilities[].comment`에 포함 | 항목과 코멘트가 어긋날 일이 없음 |
| 능력치 5개 이름 | `evasion`, `color_blend`, `texture_blend`, `exposure`, `clutter` | YOLO 박스로 계산 가능한 항목으로 교체 (4장) |
| (없음) | `found: false` | 박스를 그리지 않고 "AI가 찾지 못했어요" 화면 표시. 총점 표시 방식은 프론트가 결정 |
| (없음) | `score: null` | 레이더에서는 0으로 그리고 숫자는 `-`로 표시 |
| (없음) | 실패 응답 `error.message` | 임시 결과로 바꾸지 말고 이 문장을 보여주기 |

요청 형식(`POST /detect`, `image`)은 그대로예요. `mockResult()`는 `schemas/examples/response_found.json`의 내용으로 바꾸면 됩니다.

## 6. 정해야 할 것

| 항목 | 상태 | 언제 |
|---|---|---|
| 능력치 화면 이름 | 프론트가 `key` 기준으로 정함 | 프론트 작업 때 |
| 못 찾았을 때 화면 표현 | 프론트가 정함 (서버는 `null`) | 프론트 작업 때 |
| 능력치 계산식과 가중치 | v0 제안 기준으로 조정 | 3주차 |
| `find_difficulty` 추가 여부 | 데이터로 검증 후 | 3주차 |
| 칭호 구간 | 점수 구간별 칭호 표 | 3주차 |
| 사람이 찾은 시간(`elapsed_sec`) | 지금은 쓰지 않음 | 게임 흐름이 바뀌면 |

스키마를 바꿀 때는 `schema_version`을 올리고 세 명 모두에게 알립니다.
