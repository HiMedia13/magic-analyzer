"""의심 순간 탐지 — 손 관측 시퀀스에서 마술 기법의 '신호'를 점수화한다.

중요: 이 도구는 비밀을 단정하지 않는다. '여기서 뭔가 일어났을 가능성이 높다'는
순간을 통계적으로 짚어줄 뿐이다. 최종 판단은 사람이 그 순간을 돌려보며 한다.

탐지하는 신호
  - VANISH   : 보이던 손 개수가 줄어듦        (팜으로 숨김 / 주머니로 디치)
  - BORDER   : 손 '중심'이 화면 가장자리로 감   (랩/주머니로 이동 — 손이 통째로 빠질 때만)
  - CONTACT  : 두 손이 맞닿음                  (몰래 전달 / 로드)
  - FAST     : 손이 순간적으로 빠르게 움직임      (패스 / 던지기성 슬레이트)
  - GRAB     : 펼친 손이 갑자기 주먹으로          (코인·카드 팜)

탐지 방식: 신호를 프레임마다 가중합 → 스무딩 → **점수 봉우리(peak)를 비최대 억제(NMS)**
로 집어낸다. 클로즈업 영상은 신호가 거의 상시 켜지므로, '임계 초과 구간'을 잡으면
수십 초짜리 덩어리가 된다. 봉우리만 집으면 '여기 0:08, 0:27' 식으로 또렷해진다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hands import FrameObs

SIGNALS = ("VANISH", "BORDER", "CONTACT", "FAST", "GRAB")

# 마술 종류별 신호 가중치 (도메인 휴리스틱)
MODE_WEIGHTS = {
    "card": {"VANISH": 1.0, "BORDER": 0.7, "CONTACT": 0.6, "FAST": 1.0, "GRAB": 0.7},
    "coin": {"VANISH": 1.0, "BORDER": 0.6, "CONTACT": 1.0, "FAST": 0.5, "GRAB": 1.0},
}

SIGNAL_DESC = {
    "VANISH": "손이 화면에서 사라짐 — 팜으로 숨기거나 주머니로 디치했을 수 있음",
    "BORDER": "손이 화면 가장자리로 빠짐 — 랩/주머니/오프스크린 이동 가능",
    "CONTACT": "두 손이 맞닿음 — 몰래 물건을 전달하거나 로드했을 수 있음",
    "FAST": "손이 순간적으로 빠르게 움직임 — 패스/던지기성 슬레이트 가능",
    "GRAB": "펼친 손이 갑자기 주먹이 됨 — 코인/카드 팜 가능",
}


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    peak_sec: float
    score: float
    signals: dict[str, float] = field(default_factory=dict)  # 신호별 기여 합

    @property
    def top_signals(self) -> list[str]:
        return [s for s, v in sorted(self.signals.items(), key=lambda kv: -kv[1]) if v > 0]


def _match_velocity(prev: FrameObs | None, cur: FrameObs, fps: float) -> float:
    """현재 프레임 손들의 최대 이동속도(정규화 단위/초). 라벨로 짝짓고, 없으면 최근접."""
    if prev is None or not prev.hands or not cur.hands:
        return 0.0
    best = 0.0
    for h in cur.hands:
        same = [p for p in prev.hands if p.label == h.label] or prev.hands
        d = min(np.linalg.norm(h.center - p.center) for p in same)
        best = max(best, d * fps)
    return best


def _contact(cur: FrameObs) -> float:
    """두 손 중심 사이 거리가 가까울수록 1에 가까운 접촉 점수."""
    if len(cur.hands) < 2:
        return 0.0
    d = np.linalg.norm(cur.hands[0].center - cur.hands[1].center)
    return float(max(0.0, 1.0 - d / 0.18))  # 0.18(정규화) 이내면 접촉, 가까울수록 강함


def _center_at_border(center: np.ndarray, margin: float) -> bool:
    """손 '중심'이 가장자리 margin 이내 → 손이 통째로 빠지는 중."""
    x, y = float(center[0]), float(center[1])
    return x <= margin or y <= margin or x >= 1 - margin or y >= 1 - margin


def score_timeline(
    frames: list[FrameObs],
    fps: float,
    mode: str = "card",
    *,
    fast_thresh: float = 1.6,
    grab_drop: float = 0.45,
    border_margin: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """프레임별 신호 → 가중 점수(raw) → 스무딩(smooth)을 계산해 돌려준다.

    반환: (times, smooth_score, per_frame_signals)
    """
    weights = MODE_WEIGHTS.get(mode, MODE_WEIGHTS["card"])
    n = len(frames)
    times = np.array([f.time_sec for f in frames], dtype=np.float64)
    if n == 0:
        return times, np.zeros(0), []

    counts = [sum(1 for h in f.hands if not _center_at_border(h.center, border_margin))
              for f in frames]
    per_frame: list[dict[str, float]] = []
    prev = None
    prev_open: dict[str, float] = {}
    prev_border = False  # 직전 프레임에 '가장자리 손'이 있었는지

    for i, f in enumerate(frames):
        sig = {s: 0.0 for s in SIGNALS}

        # VANISH: 직전보다 (가장자리 아닌) 손 개수가 줄었을 때
        if i > 0 and counts[i] < counts[i - 1]:
            sig["VANISH"] = float(counts[i - 1] - counts[i])

        # BORDER: 손 중심이 가장자리로 '진입하는 순간'에만 발동 (상태 아님 → 상수 오프셋 방지)
        at_border = any(_center_at_border(h.center, border_margin) for h in f.hands)
        if at_border and not prev_border:
            sig["BORDER"] = 1.0
        prev_border = at_border

        # CONTACT
        sig["CONTACT"] = _contact(f)

        # FAST
        v = _match_velocity(prev, f, fps)
        if v > fast_thresh:
            sig["FAST"] = min(1.5, v / fast_thresh)

        # GRAB: 같은 라벨 손의 openness 급감
        cur_open: dict[str, float] = {}
        for h in f.hands:
            cur_open[h.label] = h.openness
            if h.label in prev_open:
                drop = prev_open[h.label] - h.openness
                if drop > grab_drop:
                    sig["GRAB"] = max(sig["GRAB"], min(1.5, drop / grab_drop))
        prev_open = cur_open

        per_frame.append(sig)
        prev = f

    raw = np.array([sum(weights[s] * sig[s] for s in SIGNALS) for sig in per_frame])
    # 가우시안풍 5탭 스무딩으로 단발 노이즈 억제
    if n >= 5:
        k = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        smooth = np.convolve(raw, k, mode="same")
    else:
        smooth = raw
    return times, smooth, per_frame


def detect(
    frames: list[FrameObs],
    fps: float,
    mode: str = "card",
    *,
    fast_thresh: float = 1.6,
    grab_drop: float = 0.45,
    border_margin: float = 0.06,
    score_thresh: float = 0.9,      # 봉우리가 이 점수 이상이어야 채택
    min_gap_sec: float = 1.2,       # 봉우리 사이 최소 간격(NMS)
    window_sec: float = 0.8,        # 봉우리 앞뒤로 이만큼을 '구간'으로
    max_results: int | None = None,
) -> list[Segment]:
    times, smooth, per_frame = score_timeline(
        frames, fps, mode,
        fast_thresh=fast_thresh, grab_drop=grab_drop, border_margin=border_margin,
    )
    n = len(smooth)
    if n == 0:
        return []
    weights = MODE_WEIGHTS.get(mode, MODE_WEIGHTS["card"])

    # 1) 국소 최대 후보 (임계 이상 + 양옆보다 크거나 같음)
    cand = [i for i in range(n)
            if smooth[i] >= score_thresh
            and smooth[i] >= smooth[max(0, i - 1)]
            and smooth[i] >= smooth[min(n - 1, i + 1)]]
    # 2) 비최대 억제: 점수 높은 순으로 채택, min_gap_sec 안에 이미 채택된 게 있으면 버림
    cand.sort(key=lambda i: -smooth[i])
    chosen: list[int] = []
    for i in cand:
        if all(abs(times[i] - times[j]) >= min_gap_sec for j in chosen):
            chosen.append(i)
    chosen.sort()

    # 3) 각 봉우리를 ±window_sec 창으로 만들고 그 안의 신호를 합산
    segments: list[Segment] = []
    for i in chosen:
        lo = times[i] - window_sec
        hi = times[i] + window_sec
        agg = {s: 0.0 for s in SIGNALS}
        for k in range(n):
            if lo <= times[k] <= hi:
                for s in SIGNALS:
                    agg[s] += weights[s] * per_frame[k][s]
        segments.append(Segment(
            start_sec=float(max(times[0], lo)),
            end_sec=float(min(times[-1], hi)),
            peak_sec=float(times[i]),
            score=float(smooth[i]),
            signals={s: round(v, 3) for s, v in agg.items() if v > 0},
        ))

    segments.sort(key=lambda s: -s.score)
    if max_results is not None:
        segments = segments[:max_results]
    return segments
