"""마술 분석 에이전트 (LangGraph ReAct + 도구 호출, + LangSmith 추적).

고정 파이프라인이 아니라 **도구를 호출하는 진짜 에이전트**다. 에이전트(gpt-4o)는
다음 도구를 스스로 호출하며 어디를 들여다볼지 결정한다:

  - list_suspect_moments() : 자동 탐지된 의심 순간(시각/신호/점수) 목록
  - inspect_moment(t)      : 그 시각의 프레임(직전/정점/직후)을 '비전으로' 분석해
                             손 동작 설명을 돌려줌 (영상을 들여다보는 능력)

에이전트는 목록을 보고 → 의심스러운 순간을 골라 inspect → 종합 결론을 낸다.
이미지는 inspect_moment 안의 비전 서브호출에만 들어가므로 메인 트레이스가 가볍다.

LangSmith 추적: 환경변수로 자동 + 루트에 @traceable. OpenAI 사용(Anthropic 아님).
"""

from __future__ import annotations

import base64
import os

import cv2
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langsmith import traceable

from .detect import SIGNAL_DESC

MODE_KO = {"card": "카드 마술", "coin": "동전 마술"}

AGENT_SYSTEM = (
    "당신은 클로즈업 마술(카드·동전)의 기법을 분석하는 전문가 에이전트입니다. "
    "교육·복기 목적으로 한 영상의 '비밀 동작'을 추론합니다.\n"
    "진행 방법:\n"
    "1) 먼저 list_suspect_moments 로 자동 탐지된 의심 순간을 확인합니다.\n"
    "2) 점수가 높거나 신호가 의심스러운 순간을 골라 inspect_moment(time_sec)로 "
    "들여다봅니다. (최대 6곳 정도. 같은 곳을 반복하지 마세요.)\n"
    "3) 관찰을 종합해, 이 마술 전체가 어떤 트릭이며 핵심 비밀 동작이 언제·무엇이었을지 "
    "한국어로 결론을 작성합니다.\n"
    "확신할 수 없으면 단정하지 말고 가능성으로 제시하고, 근거가 부족하면 솔직히 "
    "'판단 어려움'이라고 하세요. 마술 용어(더블리프트·팜·프렌치드롭·패스 등)를 활용하세요."
)
VISION_SYSTEM = (
    "마술 분석용입니다. 연속 프레임(직전→정점→직후)에서 두 손의 위치/모양 변화를 "
    "관찰하고, 어떤 슬레이트(팜·프렌치드롭·더블리프트·패스·비밀 전달 등)가 일어났을 "
    "법한지 한국어 2~3문장으로 설명하세요. 단정 금지, 가능성으로."
)


def _data_url(image_bgr, max_edge: int = 512, quality: int = 75) -> str:
    h, w = image_bgr.shape[:2]
    scale = max_edge / max(h, w)
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("프레임 인코딩 실패")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _maybe_enable_tracing() -> None:
    if os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"):
        os.environ.setdefault("LANGSMITH_TRACING", "true")
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", "magic-analyzer")


def _make_frame_reader(video_path: str, fps: float):
    """time_sec → 직전/정점/직후 프레임(BGR) 리스트를 주는 함수."""
    def read(time_sec: float, off: float = 0.4):
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        out = []
        for dt in (-off, 0.0, off):
            idx = max(0, min(total - 1, int(round((time_sec + dt) * fps))))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                out.append(frame)
        cap.release()
        return out
    return read


def _nearest_signals(segments: list[dict], time_sec: float) -> list[str]:
    if not segments:
        return []
    near = min(segments, key=lambda s: abs(s["peak_sec"] - time_sec))
    return near.get("top_signals", []) if abs(near["peak_sec"] - time_sec) < 1.0 else []


@traceable(name="magic-trick-agent", run_type="chain")
def analyze(video_path: str, fps: float, segments: list[dict],
           mode: str = "card", model: str = "gpt-4o",
           max_inspect: int = 6) -> dict:
    """비디오 경로 + 탐지된 의심 순간 메타로 ReAct 에이전트를 돌려 분석.

    segments: [{peak_sec, score, top_signals}, ...]
    반환: {"analyses": [{peak_sec, hypothesis, ...}], "summary": <최종 결론>}
    """
    _maybe_enable_tracing()
    if not segments:
        return {"analyses": [], "summary": "분석할 구간이 없습니다."}

    read_frames = _make_frame_reader(video_path, fps)
    inspections: list[dict] = []

    @tool
    def list_suspect_moments() -> str:
        """자동 탐지된 의심 순간 목록을 (시각/신호/점수) 텍스트로 반환한다."""
        lines = []
        for i, s in enumerate(segments):
            sig = ", ".join(s.get("top_signals", [])) or "신호 없음"
            lines.append(f"{i + 1}. {s['peak_sec']:.1f}s | 점수 {s.get('score', 0):.2f} "
                         f"| 신호: {sig}")
        return "탐지된 의심 순간:\n" + "\n".join(lines)

    @tool
    def inspect_moment(time_sec: float) -> str:
        """주어진 시각(초)의 프레임(직전/정점/직후)을 비전으로 분석해 손 동작 설명을 반환한다."""
        if len(inspections) >= max_inspect:
            return "검토 한도에 도달했습니다. 지금까지의 관찰로 결론을 작성하세요."
        frames = read_frames(float(time_sec))
        if not frames:
            return f"{time_sec:.1f}s 프레임을 읽지 못했습니다."
        sig = _nearest_signals(segments, float(time_sec))
        sig_txt = ("자동 탐지 신호: "
                   + ", ".join(f"{s}({SIGNAL_DESC[s].split(' — ')[0]})" for s in sig)) \
            if sig else "자동 탐지 신호: 특이사항 없음"
        labels = ["직전", "정점", "직후"][:len(frames)]
        content = [{"type": "text",
                    "text": f"{MODE_KO.get(mode, mode)} {time_sec:.1f}초 부근. {sig_txt}.\n"
                            f"프레임 순서: {'/'.join(labels)}."}]
        for fr in frames:
            content.append({"type": "image_url", "image_url": {"url": _data_url(fr)}})
        vision = ChatOpenAI(model=model, max_tokens=350)
        resp = vision.invoke([SystemMessage(content=VISION_SYSTEM),
                              HumanMessage(content=content)])
        desc = (resp.content or "").strip()
        inspections.append({"peak_sec": float(time_sec), "hypothesis": desc,
                            "top_signals": sig})
        return desc

    agent = create_react_agent(
        ChatOpenAI(model=model, max_tokens=1200),
        [list_suspect_moments, inspect_moment],
        prompt=AGENT_SYSTEM,
    )
    task = (f"이 {MODE_KO.get(mode, mode)} 영상의 비밀 기법을 분석하세요. "
            f"list_suspect_moments로 의심 순간을 확인하고, 의심스러운 곳을 "
            f"inspect_moment로 들여다본 뒤 전체 트릭과 핵심 비밀 동작을 종합 결론으로 쓰세요.")
    result = agent.invoke({"messages": [("user", task)]},
                          config={"recursion_limit": 2 * max_inspect + 8})
    summary = ""
    for m in reversed(result["messages"]):
        if getattr(m, "type", "") == "ai" and m.content:
            summary = m.content if isinstance(m.content, str) else str(m.content)
            break

    analyses = sorted(inspections, key=lambda a: -_score_for(segments, a["peak_sec"]))
    for a in analyses:
        a["score"] = round(_score_for(segments, a["peak_sec"]), 3)
    return {"analyses": analyses, "summary": summary}


def _score_for(segments: list[dict], time_sec: float) -> float:
    if not segments:
        return 0.0
    near = min(segments, key=lambda s: abs(s["peak_sec"] - time_sec))
    return float(near.get("score", 0.0)) if abs(near["peak_sec"] - time_sec) < 1.0 else 0.0
