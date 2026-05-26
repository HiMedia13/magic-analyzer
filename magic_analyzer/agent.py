"""LangGraph 기반 마술 분석 에이전트 (+ LangSmith 추적).

단발 LLM 호출이 아니라, 그래프 형태의 다단계 분석으로 구성한다:

    START ─(구간별 fan-out)→ analyze_one ─→ synthesize ─→ END

  - analyze_one : 의심 구간마다 직전/정점/직후 3프레임을 비전 모델로 분석(병렬)
  - synthesize  : 구간별 가설을 모아 '전체 트릭이 무엇을 하는지' 추정

LangSmith 추적은 환경변수(LANGSMITH_TRACING / LANGSMITH_API_KEY / LANGSMITH_PROJECT)로
자동 활성화되며, 루트 실행 함수에 @traceable을 달아 한 트레이스로 묶는다.

OPENAI_API_KEY 필요. (Anthropic 아님 — OpenAI 사용)
"""

from __future__ import annotations

import base64
import operator
import os
from typing import Annotated, TypedDict

import cv2
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langsmith import traceable

from .detect import SIGNAL_DESC

MODE_KO = {"card": "카드 마술", "coin": "동전 마술"}

ANALYZE_SYSTEM = (
    "당신은 클로즈업 마술(카드·동전) 기법 분석 전문가입니다. 교육·복기 목적으로, "
    "마술사가 '비밀 동작'을 했을 가능성이 높은 순간의 연속 프레임(직전→정점→직후)을 봅니다. "
    "손의 위치/모양 변화를 근거로 어떤 슬레이트(더블리프트, 팜, 프렌치드롭, 패스, 비밀 전달 등)가 "
    "일어났을 법한지 추론하세요. 확신 없으면 단정하지 말고 가능성으로, 근거 부족하면 "
    "'판단 어려움'이라고 정직하게. 한국어 3~4문장."
)
SYNTH_SYSTEM = (
    "당신은 마술 루틴 분석가입니다. 아래는 한 영상에서 자동 탐지된 여러 의심 구간과 "
    "구간별 기법 추론입니다. 이를 종합해 '이 마술 전체가 어떤 트릭이며 핵심 비밀 동작이 "
    "언제·무엇이었을지'를 한국어 4~6문장으로 정리하세요. 단정 대신 가능성으로 제시하세요."
)


# ---------- 그래프 상태 ----------
class State(TypedDict):
    mode: str
    model: str
    segments: list[dict]                       # fan-out 입력
    analyses: Annotated[list[dict], operator.add]  # 병렬 결과 누적
    summary: str


class SegTask(TypedDict):
    mode: str
    model: str
    index: int
    segment: dict


def _data_url(image_bgr, max_edge: int = 512, quality: int = 75) -> str:
    """프레임을 작게 리사이즈+JPEG 인코딩해 data URL로. (비전 비용·트레이스 크기 절감)

    이미지 바이트가 LangGraph 상태/트레이스에 들어가므로 너무 크면 LangSmith 20MB
    ingest 한도를 넘는다. 손 위치 판단에는 512px면 충분하다.
    """
    h, w = image_bgr.shape[:2]
    scale = max_edge / max(h, w)
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("프레임 인코딩 실패")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _signals_text(top_signals: list[str]) -> str:
    return ", ".join(f"{s}({SIGNAL_DESC[s].split(' — ')[0]})" for s in top_signals) \
        or "특이 신호 없음"


# ---------- 노드 ----------
def _analyze_one(task: SegTask) -> dict:
    seg = task["segment"]
    urls = seg.get("image_urls") or []
    if not urls:
        return {"analyses": [{**_meta(seg, task["index"]),
                              "hypothesis": "(프레임 없음 — 추론 불가)"}]}

    labels = ["직전", "정점", "직후"][:len(urls)]
    text = (
        f"종류: {MODE_KO.get(task['mode'], task['mode'])}\n"
        f"의심 시각: {seg['peak_sec']:.1f}초 부근\n"
        f"자동 탐지 신호: {_signals_text(seg.get('top_signals', []))}\n"
        f"프레임 순서: {'/'.join(labels)}. 여기서 무슨 일이 일어났을지 추론하세요."
    )
    content = [{"type": "text", "text": text}]
    for url in urls:
        content.append({"type": "image_url", "image_url": {"url": url}})

    llm = ChatOpenAI(model=task["model"], max_tokens=400)
    resp = llm.invoke([SystemMessage(content=ANALYZE_SYSTEM),
                       HumanMessage(content=content)])
    return {"analyses": [{**_meta(seg, task["index"]),
                          "hypothesis": (resp.content or "").strip()}]}


def _synthesize(state: State) -> dict:
    analyses = sorted(state["analyses"], key=lambda a: -a.get("score", 0))
    if not analyses:
        return {"summary": "분석할 구간이 없습니다."}
    lines = [f"- {a['peak_sec']:.1f}s (점수 {a.get('score', 0):.2f}): {a['hypothesis']}"
             for a in analyses]
    text = (f"종류: {MODE_KO.get(state['mode'], state['mode'])}\n"
            f"구간별 추론:\n" + "\n".join(lines))
    llm = ChatOpenAI(model=state["model"], max_tokens=500)
    resp = llm.invoke([SystemMessage(content=SYNTH_SYSTEM),
                       HumanMessage(content=text)])
    return {"summary": (resp.content or "").strip()}


def _meta(seg: dict, index: int) -> dict:
    return {"index": index, "peak_sec": seg["peak_sec"],
            "score": seg.get("score", 0.0), "top_signals": seg.get("top_signals", [])}


def _fan_out(state: State):
    return [Send("analyze_one", {"mode": state["mode"], "model": state["model"],
                                 "index": i, "segment": seg})
            for i, seg in enumerate(state["segments"])]


def _build_graph():
    g = StateGraph(State)
    g.add_node("analyze_one", _analyze_one)
    g.add_node("synthesize", _synthesize)
    g.add_conditional_edges(START, _fan_out, ["analyze_one"])
    g.add_edge("analyze_one", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


GRAPH = _build_graph()  # 컴파일은 1회 (재사용 가능)


def _maybe_enable_tracing() -> None:
    """LANGSMITH_API_KEY가 있으면 추적을 켜고 프로젝트 기본값을 설정."""
    if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "magic-analyzer")


def _redact_inputs(inputs: dict) -> dict:
    """트레이스에 기록할 입력에서 raw 이미지 픽셀을 제거(프레임 수만 남김).

    @traceable은 함수 인자를 그대로 직렬화하는데, segments의 numpy 프레임을 그대로
    두면 트레이스가 수십 MB로 불어 LangSmith ingest 한도(20MB)를 넘는다.
    """
    segs = inputs.get("segments") or []
    red = [{"peak_sec": s.get("peak_sec"), "score": s.get("score"),
            "top_signals": s.get("top_signals"),
            "n_frames": len(s.get("images") or [])} for s in segs]
    return {**{k: v for k, v in inputs.items() if k != "segments"}, "segments": red}


@traceable(name="magic-trick-agent", run_type="chain", process_inputs=_redact_inputs)
def analyze(segments: list[dict], mode: str = "card",
            model: str = "gpt-4o") -> dict:
    """의심 구간 리스트(각각 peak_sec/score/top_signals/images[BGR])를 받아
    LangGraph 에이전트로 분석하고 {analyses, summary}를 반환."""
    _maybe_enable_tracing()
    if not segments:
        return {"analyses": [], "summary": "분석할 구간이 없습니다."}
    # 프레임(BGR)을 미리 작은 data URL 문자열로 인코딩해 상태에 넣는다.
    # (raw 픽셀을 상태에 두면 트레이스가 LangSmith 한도를 초과한다)
    enc = []
    for seg in segments:
        enc.append({
            "peak_sec": seg["peak_sec"],
            "score": seg.get("score", 0.0),
            "top_signals": seg.get("top_signals", []),
            "image_urls": [_data_url(im) for im in (seg.get("images") or [])],
        })
    result = GRAPH.invoke({
        "mode": mode, "model": model, "segments": enc,
        "analyses": [], "summary": "",
    })
    analyses = sorted(result.get("analyses", []), key=lambda a: -a.get("score", 0))
    return {"analyses": analyses, "summary": result.get("summary", "")}
