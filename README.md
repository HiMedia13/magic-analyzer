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

```powershell
# 기본 분석 (리포트만)
python main.py trick.mp4 --mode card

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
| `--annotate` | 손 관절+배너 표시된 `annotated.mp4` 저장 |
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
- `annotated.mp4` — (옵션) 관절·의심구간 표시 영상
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

## LLM 추론 (OpenAI, 옵션)
`--llm`을 켜면 상위 의심 구간마다 **직전→정점→직후 3프레임**을 OpenAI 비전 모델
(`gpt-4o`)에 보내 "여기서 어떤 기법이 일어났을 법한지"를 한국어로 추론합니다.
한 장이 아니라 3프레임을 보내 슬레이트의 *움직임 흐름*을 보여주는 게 핵심입니다.

```powershell
$env:OPENAI_API_KEY = "sk-..."     # 키 설정 (필수)
python main.py samples/coin_cu.mp4 --mode coin --llm --llm-top 5
```

결과는 콘솔 + `out/llm.txt` + `out/llm.json`에 저장됩니다. 키가 없으면 분석은
정상 진행되고 LLM 단계만 건너뜁니다(친절한 안내 출력). 비밀을 단정하지 않고
"가능성"으로 제시하도록 프롬프트가 설계돼 있습니다.

> 참고: 이 기능은 Anthropic이 아니라 **OpenAI SDK**(`openai`)를 사용합니다.

## 개발용 튜닝
`scripts/tune.py` 는 손 추적 결과를 pickle로 캐시한 뒤 detect 파라미터만 바꿔가며
빠르게 비교한다(영상당 추적 1회). 점수 분포(`--hist`)도 출력한다.
```powershell
python scripts/tune.py samples/coin.mp4 --mode coin --score-thresh 1.0 --hist
```

## 로드맵
- [ ] 동전/카드 객체 자체 추적(사라짐 직접 감지)
- [x] 의심 프레임을 멀티모달 LLM(OpenAI)에 넘겨 "무슨 일이 일어났는지" 추론
- [ ] 간단한 웹 UI

## 면책
교육·연습 복기 목적의 보조 도구입니다.
