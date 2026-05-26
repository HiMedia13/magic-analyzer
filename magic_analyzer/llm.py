"""의심 구간 프레임을 OpenAI 비전 모델에 넘겨 '무슨 일이 일어났는지' 추론한다.

각 의심 구간마다 직전/정점/직후 3장을 함께 보내 '움직임의 흐름'을 보여준다.
한 장만 보내면 슬레이트의 핵심(빠른 손동작)이 안 보이기 때문이다.

OPENAI_API_KEY 환경변수가 필요하다. claude가 아니라 OpenAI SDK를 쓴다.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import cv2

from .detect import SIGNAL_DESC, Segment

MODE_KO = {"card": "카드 마술", "coin": "동전 마술"}

SYSTEM_PROMPT = (
    "당신은 클로즈업 마술(카드·동전)의 기법을 분석하는 전문가입니다. "
    "교육·연습 복기 목적으로, 마술사가 '비밀 동작'을 했을 가능성이 높은 순간의 "
    "연속 프레임(직전→정점→직후)을 받습니다. 손의 위치/모양 변화를 근거로, "
    "여기서 어떤 슬레이트나 기법(예: 더블리프트, 팜, 프렌치드롭, 패스, 비밀 전달 등)이 "
    "일어났을 법한지 추론하세요. 확신할 수 없으면 단정하지 말고 가능성으로 제시하고, "
    "근거가 부족하면 솔직히 '판단 어려움'이라고 답하세요. 한국어로 3~4문장 이내."
)


@dataclass
class FrameTriplet:
    """한 구간의 직전/정점/직후 프레임(BGR)."""
    before: "cv2.typing.MatLike | None"
    peak: "cv2.typing.MatLike | None"
    after: "cv2.typing.MatLike | None"

    def present(self) -> list[tuple[str, "cv2.typing.MatLike"]]:
        out = []
        for label, img in (("직전", self.before), ("정점", self.peak), ("직후", self.after)):
            if img is not None:
                out.append((label, img))
        return out


def _to_data_url(image_bgr) -> str:
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("프레임 JPEG 인코딩 실패")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


class LLMInferencer:
    def __init__(self, model: str = "gpt-4o", api_key: str | None = None) -> None:
        # 지연 임포트 — --llm을 안 쓰면 openai가 없어도 동작하도록
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._model = model

    def infer(self, seg: Segment, triplet: FrameTriplet, mode: str) -> str:
        frames = triplet.present()
        if not frames:
            return "(프레임 없음 — 추론 불가)"

        signals = ", ".join(f"{s}({SIGNAL_DESC[s].split(' — ')[0]})"
                            for s in seg.top_signals) or "특이 신호 없음"
        text = (
            f"종류: {MODE_KO.get(mode, mode)}\n"
            f"의심 시각: {seg.peak_sec:.1f}초 부근\n"
            f"자동 탐지 신호: {signals}\n"
            f"아래 프레임은 순서대로 {'/'.join(l for l, _ in frames)} 입니다. "
            f"여기서 무슨 일이 일어났을지 추론해 주세요."
        )
        content: list[dict] = [{"type": "text", "text": text}]
        for _, img in frames:
            content.append({"type": "image_url",
                            "image_url": {"url": _to_data_url(img)}})

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=400,
        )
        return (resp.choices[0].message.content or "").strip()
