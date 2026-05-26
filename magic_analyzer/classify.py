"""영상이 '카드 마술'인지 '동전 마술'인지 자동 분류 (OpenAI 비전).

영상 전반에서 프레임 몇 장을 고르게 뽑아 gpt-4o에게 어떤 소품을 쓰는지 묻고
card / coin 중 하나로 판별한다. 손 궤적만으론 카드/동전 구분이 안 되므로, 모드를
자동으로 정하려면 '무엇이 보이는지'를 봐야 한다. OPENAI_API_KEY 필요.
"""

from __future__ import annotations

import base64

import cv2

CLASSIFY_SYSTEM = (
    "당신은 마술 영상 분류기입니다. 주어진 프레임들을 보고 이 마술이 주로 무엇을 다루는지 "
    "판단하세요. 카드(트럼프 카드)를 주로 쓰면 'card', 동전을 주로 쓰면 'coin'으로 답합니다. "
    "둘 다 보이면 더 두드러진 쪽을 고르세요. 반드시 'card' 또는 'coin' 한 단어로만 답하세요."
)


def _data_url(image_bgr, max_edge: int = 384) -> str:
    h, w = image_bgr.shape[:2]
    scale = max_edge / max(h, w)
    if scale < 1.0:
        image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _sample_frames(video_path: str, n: int) -> list:
    """영상 전반에서 n장을 고르게 샘플(앞뒤 5%는 제외)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    if total <= 0:
        cap.release()
        return frames
    lo, hi = int(total * 0.05), int(total * 0.95)
    step = max(1, (hi - lo) // max(1, n))
    for k in range(n):
        idx = min(hi, lo + k * step)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if ok:
            frames.append(img)
    cap.release()
    return frames


def classify_video(video_path: str, n: int = 5, model: str = "gpt-4o") -> str:
    """영상을 'card' 또는 'coin'으로 분류. 실패 시 예외를 올린다(호출측에서 폴백)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    frames = _sample_frames(video_path, n)
    if not frames:
        raise RuntimeError("프레임을 읽지 못했습니다.")
    content = [{"type": "text",
                "text": "이 프레임들을 보고 card 또는 coin 한 단어로만 답하세요."}]
    for img in frames:
        content.append({"type": "image_url", "image_url": {"url": _data_url(img)}})

    llm = ChatOpenAI(model=model, max_tokens=5)
    resp = llm.invoke([SystemMessage(content=CLASSIFY_SYSTEM),
                       HumanMessage(content=content)])
    text = (resp.content or "").strip().lower()
    if "coin" in text or "동전" in text:
        return "coin"
    return "card"  # 기본/모호하면 card
