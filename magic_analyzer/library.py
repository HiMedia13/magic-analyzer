"""예시 라이브러리 매칭 (few-shot) — 기법 시연의 '손 궤적 시그니처'를 모아두고,
의심 순간의 궤적을 코사인 유사도로 비교해 가장 닮은 기법을 찾는다.

큰 학습 데이터 없이 '영상에서 배운 예시'와 비교하는 방식. 시그니처는 위치·크기에
불변하도록 손목 기준으로 정규화하고, 시간 차이를 흡수하도록 고정 길이로 리샘플한다.

한계: 튜토리얼 시연 ≠ 실제 공연(속도·각도 차이), 라이브러리가 작으면 커버리지 제한.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .hands import HandTracker
from .video import open_video

LIBRARY_PATH = Path(__file__).resolve().parent.parent / "library" / "signatures.json"

WINDOW_SEC = 0.6      # 순간 앞뒤로 볼 시간
N_RESAMPLE = 12       # 시퀀스를 이 길이로 리샘플
_DIM = N_RESAMPLE * 21 * 2  # 시그니처 차원


def _primary_vec(obs) -> np.ndarray | None:
    """프레임에서 주(가장 큰) 손의 손목기준·크기정규화 랜드마크 벡터(42,)."""
    if not obs.hands:
        return None
    h = max(obs.hands, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    lm = h.landmarks.astype(np.float64)           # (21,2)
    wrist = lm[0]
    scale = np.linalg.norm(lm[9] - wrist) + 1e-6  # 손목~중지뿌리 = 손 크기
    return ((lm - wrist) / scale).reshape(-1)     # (42,)


def _signature(frame_vecs: list[np.ndarray | None]) -> np.ndarray | None:
    """프레임별 벡터 시퀀스 → 고정 길이 리샘플 → 평탄화 → L2 정규화 시그니처."""
    present = [v for v in frame_vecs if v is not None]
    if len(present) < 3:
        return None
    arr = np.stack(present)                        # (T,42)
    idx = np.linspace(0, len(arr) - 1, N_RESAMPLE).round().astype(int)
    sig = arr[idx].reshape(-1)                     # (N_RESAMPLE*42,)
    n = np.linalg.norm(sig) + 1e-9
    return sig / n


def signature_from_video(video_path: str, time_sec: float,
                         window: float = WINDOW_SEC) -> np.ndarray | None:
    """영상의 특정 시각 주변 손 궤적을 시그니처로 추출."""
    import cv2
    cap, meta = open_video(video_path)
    fps = meta.fps
    vecs: list[np.ndarray | None] = []
    try:
        start_idx = max(0, int((time_sec - window) * fps))
        end_idx = int((time_sec + window) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        with HandTracker() as tracker:
            idx = start_idx
            while idx <= end_idx:
                ok, img = cap.read()
                if not ok:
                    break
                obs = tracker.process(idx, idx / fps, img)
                vecs.append(_primary_vec(obs))
                idx += 1
    finally:
        cap.release()
    return _signature(vecs)


# ---------- 라이브러리 입출력 ----------
def load_library(path: Path = LIBRARY_PATH) -> list[dict]:
    """[{technique, name_ko, vec:[...]}, ...]"""
    if not Path(path).exists():
        return []
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []


def save_entry(technique: str, name_ko: str, sig: np.ndarray,
               source: str = "", path: Path = LIBRARY_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lib = load_library(path)
    lib.append({"technique": technique, "name_ko": name_ko,
                "vec": [round(float(x), 5) for x in sig], "source": source})
    path.write_text(json.dumps(lib, ensure_ascii=False), encoding="utf-8")


def match(sig: np.ndarray, library: list[dict], k: int = 3) -> list[dict]:
    """시그니처를 라이브러리와 코사인 유사도 비교 → 기법별 최고 유사도 상위 k."""
    if sig is None or not library:
        return []
    best: dict[str, dict] = {}
    for e in library:
        v = np.asarray(e["vec"], dtype=np.float64)
        if v.shape[0] != sig.shape[0]:
            continue
        cos = float(np.dot(sig, v))  # 둘 다 L2 정규화돼 있으므로 내적=코사인
        key = e["technique"]
        if key not in best or cos > best[key]["similarity"]:
            best[key] = {"technique": e["technique"], "name_ko": e.get("name_ko", e["technique"]),
                         "similarity": round(cos, 3)}
    return sorted(best.values(), key=lambda x: -x["similarity"])[:k]
