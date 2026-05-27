"""MediaPipe 모델 파일 확보 — 없으면 공식 저장소에서 자동 다운로드한다."""

from __future__ import annotations

import urllib.request
from pathlib import Path

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# 모델은 패키지 옆 models/ 폴더에 캐시 (git에는 올리지 않음)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def ensure_hand_model() -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dst = MODELS_DIR / "hand_landmarker.task"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    print(f"      손 모델 다운로드 중... ({HAND_MODEL_URL.split('/')[-1]})")
    # 임시 파일로 받은 뒤 원자적으로 교체 — 중단되면 .part만 남고 손상된 모델이
    # dst 자리에 남지 않는다(다음 실행에서 size>0 검사를 통과해버리는 문제 방지).
    tmp = dst.with_name(dst.name + ".part")
    try:
        urllib.request.urlretrieve(HAND_MODEL_URL, tmp)
        tmp.replace(dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dst
