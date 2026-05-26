# magic-analyzer 🃏🪙

카드/동전 마술 **영상**을 넣으면, 손을 추적해 **"수상한 순간"**을 타임라인으로 짚어주는 분석 도구.

> ⚠️ **솔직한 한계**
> 마술은 사람의 지각을 속이도록 설계돼 있어, 영상 한 개로 비밀을 **확정**하는 것은 불가능합니다.
> 이 도구는 비밀을 단정하지 않고, **통계적으로 의심스러운 구간**(손이 사라짐·두 손이 닿음·급동작·주먹쥠)을
> 찾아 사람이 다시 돌려보도록 돕는 **보조 도구**입니다.

## 무엇을 탐지하나

| 신호 | 의미(가능성) |
|------|------|
| `VANISH` | 보이던 손이 사라짐 — 팜으로 숨김 / 주머니로 디치 |
| `BORDER` | 손이 화면 가장자리로 빠짐 — 랩/주머니/오프스크린 이동 |
| `CONTACT` | 두 손이 맞닿음 — 몰래 전달 / 로드 |
| `FAST` | 손이 순간적으로 빠르게 움직임 — 패스/슬레이트 |
| `GRAB` | 펼친 손이 갑자기 주먹 — 코인/카드 팜 |

`--mode card` 와 `--mode coin` 은 위 신호의 가중치가 다릅니다.

## 설치

Python 3.12 권장 (MediaPipe 호환).

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 사용법

입력은 **로컬 파일 경로** 또는 **YouTube URL** 둘 다 됩니다. URL이면 yt-dlp로
자동 다운로드(360p, ffmpeg 불필요)한 뒤 분석하며, 같은 영상은 `downloads/`에 캐시됩니다.

```powershell
# 기본 분석 (리포트만)
python main.py trick.mp4 --mode card

# YouTube URL 직접 분석
python main.py "https://youtu.be/VIDEO_ID" --mode coin --llm

# 손 관절+의심구간이 표시된 영상과 의심 프레임 이미지까지 저장
python main.py trick.mp4 --mode coin --annotate --save-frames 5

# 더 민감하게 (더 많은 구간 잡기)
python main.py trick.mp4 --mode card --score-thresh 0.4
```

### 옵션
| 옵션 | 설명 |
|------|------|
| `--mode {card,coin}` | 마술 종류 (기본 card) |
| `--out DIR` | 결과 폴더 (기본 `out`) |
| `--annotate` | 손 관절+배너 표시된 `annotated.webm` 저장 (브라우저 재생 가능) |
| `--save-frames K` | 상위 K개 의심 순간 정점 프레임 이미지 저장 |
| `--stride N` | N프레임마다 1장만 분석(속도용) |
| `--score-thresh F` | 봉우리 채택 임계값 (기본 0.9, 낮출수록 많이 잡음) |
| `--min-gap S` | 의심 순간 사이 최소 간격 초 (기본 1.2) |
| `--window S` | 봉우리 앞뒤로 구간에 포함할 초 (기본 0.8) |
| `--max-results N` | 최대 의심 순간 개수 (기본: 제한 없음) |
| `--llm` | 상위 의심 구간을 **OpenAI 비전 모델로 추론** (`OPENAI_API_KEY` 필요) |
| `--llm-model M` | OpenAI 비전 모델명 (기본 `gpt-4o`) |
| `--llm-top K` | LLM으로 추론할 상위 구간 개수 (기본 5) |

## 좋은 결과를 위한 팁 (실측 기반)
- **클로즈업 + 단독 출연** 영상에서 가장 잘 작동한다. 손이 화면을 충분히 채워야
  MediaPipe가 두 손을 안정적으로 추적한다. (카드 튜토리얼/테이블 클로즈업이 이상적)
- **무대 와이드샷 + 관객 동석**은 부정확하다 — 엉뚱한 사람 손을 잡거나 작은 손을 놓친다.
- 너무 적게/많이 잡히면 `--score-thresh`로 조절 (기본 0.9 ≈ 상위 5% 순간).
- **해상도는 거의 무관**하다. MediaPipe 손 검출기가 프레임을 ~192px로 다운스케일하므로,
  같은 영상을 360p↔1080p로 비교해도 손 검출률은 사실상 동일(80.5%↔79.9%)하고 1080p는
  24% 느리기만 했다. 화질보다 **프레이밍(클로즈업)**이 결과를 좌우한다.

## 출력물 (`out/`)
- `report.txt` — 사람이 읽는 타임라인 리포트
- `report.json` — 구간/점수/신호 상세 (다른 도구·LLM에 넘기기 좋음)
- `annotated.webm` — (옵션) 관절·의심구간 표시 영상 (VP8, 브라우저 재생 가능)
- `suspect_NN_*.jpg` — (옵션) 의심 구간 정점 프레임
- `llm.txt` / `llm.json` — (옵션 `--llm`) 구간별 OpenAI 비전 추론 결과

## 동작 원리
1. **손 추적** — MediaPipe Tasks(HandLandmarker, VIDEO 모드)로 프레임마다 손 21개 관절을
   정규화 좌표로 추출
2. **신호 계산** — 손 개수 변화(VANISH) / 가장자리 *진입 이벤트*(BORDER) / 두 손 거리(CONTACT)
   / 이동속도(FAST) / 펼침정도 급감(GRAB)
3. **점수화·봉우리 검출** — 모드별 가중합 → 스무딩 → **점수 봉우리를 비최대 억제(NMS)**로
   집어 `봉우리 ±window` 창으로 보고. 클로즈업은 신호가 상시 켜지므로 '임계 초과 구간'을
   잡으면 수십 초 덩어리가 된다 — 봉우리만 집어 또렷한 순간을 준다.

## 웹 UI
브라우저에서 URL 입력/파일 업로드 → 옵션 선택 → 결과(마킹 영상·타임라인·의심
프레임·AI 설명)를 한 화면에서 봅니다. 검증된 CLI를 백엔드로 그대로 재사용합니다.

```powershell
python webapp/app.py
# 브라우저에서 http://127.0.0.1:5000 접속
```

분석은 백그라운드로 돌아가고 화면이 진행상태를 폴링합니다. `AI 설명`을 켜려면
`OPENAI_API_KEY`(.env)가 필요합니다.

## 에이전트 분석 (LangGraph ReAct + 도구 호출 + LangSmith, 옵션)
`--llm`을 켜면 고정 파이프라인이 아니라 **도구를 스스로 호출하는 ReAct 에이전트**
(`langgraph.prebuilt.create_react_agent`, `gpt-4o`)가 분석합니다. 에이전트가 쓰는 도구:

- **`list_suspect_moments()`** — 자동 탐지된 의심 순간(시각/신호/점수) 목록
- **`inspect_moment(time_sec)`** — 그 시각의 프레임(직전/정점/직후)을 **비전으로 들여다보는**
  도구. 에이전트가 *어디를 볼지 직접 결정*해 호출합니다.

에이전트는 목록 확인 → 의심스러운 순간을 골라 inspect → 종합 결론(전체 트릭 추정)을 냅니다.
도구 호출은 LangSmith 트레이스에 그대로 기록됩니다(`list_suspect_moments`, `inspect_moment`).
이미지는 inspect 도구 안의 비전 서브호출에만 들어가 메인 트레이스가 가볍습니다.

실행 과정은 **LangSmith**에 자동 기록됩니다(키가 있을 때). 키는 `.env`로 줍니다
(`.env.example` 참고):

```dotenv
OPENAI_API_KEY=sk-...           # 필수
LANGSMITH_API_KEY=lsv2_...      # 선택(추적). 있으면 자동으로 추적 on + 프로젝트 magic-analyzer
```

```powershell
python main.py samples/coin_cu.mp4 --mode coin --llm --llm-top 5
```

결과는 콘솔 + `out/llm.txt` + `out/llm.json`(구간별 추론 + **전체 트릭 추정**)에 저장됩니다.
OpenAI 키가 없으면 분석은 정상 진행되고 에이전트 단계만 건너뜁니다.

> 참고: LLM은 Anthropic이 아니라 **OpenAI**(`langchain-openai`)를 사용합니다.

## 개발용 튜닝
`scripts/tune.py` 는 손 추적 결과를 pickle로 캐시한 뒤 detect 파라미터만 바꿔가며
빠르게 비교한다(영상당 추적 1회). 점수 분포(`--hist`)도 출력한다.
```powershell
python scripts/tune.py samples/coin.mp4 --mode coin --score-thresh 1.0 --hist
```

## 로드맵
- [ ] 동전/카드 객체 자체 추적(사라짐 직접 감지)
- [x] 의심 프레임을 멀티모달 LLM(OpenAI)에 넘겨 "무슨 일이 일어났는지" 추론
- [x] 간단한 웹 UI (Flask, `webapp/app.py`)

## 면책
교육·연습 복기 목적의 보조 도구입니다.
