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
    "교육·복기 목적으로 한 영상의 '비밀 동작'을 추론하고 '기법 자체를 자세히 설명'합니다.\n"
    "진행 방법:\n"
    "1) list_suspect_moments 로 자동 탐지된 의심 순간을 확인합니다.\n"
    "2) 의심스러운 순간을 골라 inspect_moment(time_sec)로 들여다봅니다. "
    "(최대 6곳, 같은 곳 반복 금지.) 또한 match_technique(time_sec)로 '예시 "
    "라이브러리'와의 데이터 기반 유사도(어떤 기법과 닮았는지)도 확인하세요.\n"
    "3) 의심되는 기법마다 explain_technique(기법명)을 호출해 정확한 설명과 참고 "
    "영상 링크를 확보합니다. (예: explain_technique('프렌치 드롭'))\n"
    "4) 다음을 '자세히' 담은 결론을 한국어로 작성합니다:\n"
    "   - 이 마술이 무엇을 보여주는 트릭인지\n"
    "   - 사용된 기법 각각이 '어떻게 작동하는지' (동작 원리)\n"
    "   - 영상의 어느 순간·어떤 손동작이 그 근거인지 (관찰된 단서)\n"
    "확신 없으면 단정 말고 가능성으로, 근거 부족하면 '판단 어려움'이라고 하세요."
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

    from .library import load_library, match, signature_from_video
    library = load_library()

    read_frames = _make_frame_reader(video_path, fps)
    inspections: list[dict] = []
    techniques_found: list[dict] = []
    matches: list[dict] = []  # match_technique 호출 결과(데이터 기반 유사도)

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

    @tool
    def explain_technique(technique: str) -> str:
        """의심되는 마술 기법의 자세한 설명(작동 원리)과 참고 튜토리얼 영상 링크를 반환한다.
        기법명을 한글 또는 영어로 입력한다. 예: '프렌치 드롭', 'double lift'."""
        from .techniques import entry_to_dict, lookup, search_url
        e = lookup(technique)
        if e:
            d = entry_to_dict(e)
            if not any(t["name_en"] == d["name_en"] for t in techniques_found):
                techniques_found.append(d)
            return (f"{d['name_ko']} ({d['name_en']}, {d['type']})\n"
                    f"작동 원리: {d['desc']}\n관찰 단서: {d['cues']}\n"
                    f"참고 영상: {d['reference_url']}")
        url = search_url(technique)
        if not any(t["name_en"] == technique for t in techniques_found):
            techniques_found.append({"name_ko": technique, "name_en": technique,
                                     "type": "", "desc": "(용어집에 없음 — 일반 지식 기반)",
                                     "cues": "", "reference_url": url})
        return f"'{technique}'은 용어집에 없습니다. 일반 지식으로 설명하고, 참고 영상: {url}"

    @tool
    def match_technique(time_sec: float) -> str:
        """주어진 시각의 손 궤적을 '기법 예시 라이브러리'와 비교해 가장 닮은 기법과
        유사도(0~1)를 반환한다. 비전 관찰과 별개의 데이터 기반 단서."""
        if not library:
            return "기법 예시 라이브러리가 비어 있습니다(등록된 예시 없음)."
        sig = signature_from_video(video_path, float(time_sec))
        if sig is None:
            return f"{time_sec:.1f}s의 손 궤적을 얻지 못했습니다(손 미검출)."
        res = match(sig, library, k=3)
        if not res:
            return "유사한 기법을 찾지 못했습니다."
        matches.append({"time_sec": round(float(time_sec), 2), "results": res})
        return "라이브러리 매칭(유사도 0~1): " + ", ".join(
            f"{r['name_ko']} {r['similarity']:.2f}" for r in res)

    agent = create_react_agent(
        ChatOpenAI(model=model, max_tokens=1500),
        [list_suspect_moments, inspect_moment, explain_technique, match_technique],
        prompt=AGENT_SYSTEM,
    )
    task = (f"이 {MODE_KO.get(mode, mode)} 영상의 비밀 기법을 분석하세요. "
            f"list_suspect_moments로 의심 순간을 확인하고, 의심스러운 곳을 "
            f"inspect_moment로 들여다본 뒤 전체 트릭과 핵심 비밀 동작을 종합 결론으로 쓰세요.")
    result = agent.invoke({"messages": [("user", task)]},
                          config={"recursion_limit": 4 * max_inspect + 16})
    summary = ""
    for m in reversed(result["messages"]):
        if getattr(m, "type", "") == "ai" and m.content:
            summary = m.content if isinstance(m.content, str) else str(m.content)
            break

    analyses = sorted(inspections, key=lambda a: -_score_for(segments, a["peak_sec"]))
    for a in analyses:
        a["score"] = round(_score_for(segments, a["peak_sec"]), 3)
    return {"analyses": analyses, "summary": summary,
            "techniques": techniques_found, "matches": matches}


def _score_for(segments: list[dict], time_sec: float) -> float:
    if not segments:
        return 0.0
    near = min(segments, key=lambda s: abs(s["peak_sec"] - time_sec))
    return float(near.get("score", 0.0)) if abs(near["peak_sec"] - time_sec) < 1.0 else 0.0
