"""의심 순간 탐지 — 손 관측 시퀀스에서 마술 기법의 '신호'를 점수화한다.

중요: 이 도구는 비밀을 단정하지 않는다. '여기서 뭔가 일어났을 가능성이 높다'는
구간을 통계적으로 짚어줄 뿐이다. 최종 판단은 사람이 그 구간을 돌려보며 한다.

탐지하는 신호
  - VANISH   : 보이던 손 개수가 줄어듦        (팜으로 숨김 / 주머니로 디치)
  - BORDER   : 손이 화면 가장자리로 빠짐        (랩/주머니로 이동)
  - CONTACT  : 두 손이 맞닿음                  (몰래 전달 / 로드)
  - FAST     : 손이 순간적으로 빠르게 움직임      (패스 / 던지기성 슬레이트)
  - GRAB     : 펼친 손이 갑자기 주먹으로          (코인·카드 팜)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hands import FrameObs

SIGNALS = ("VANISH", "BORDER", "CONTACT", "FAST", "GRAB")

# 마술 종류별 신호 가중치 (도메인 휴리스틱)
MODE_WEIGHTS = {
    "card": {"VANISH": 1.0, "BORDER": 0.8, "CONTACT": 0.6, "FAST": 1.0, "GRAB": 0.7},
    "coin": {"VANISH": 1.0, "BORDER": 0.7, "CONTACT": 1.0, "FAST": 0.5, "GRAB": 1.0},
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
        return [s for s, _ in sorted(self.signals.items(), key=lambda kv: -kv[1]) if _ > 0]


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
    # 0.18(정규화) 이내면 접촉으로 간주, 가까울수록 강함
    return float(max(0.0, 1.0 - d / 0.18))


def detect(
    frames: list[FrameObs],
    fps: float,
    mode: str = "card",
    *,
    fast_thresh: float = 1.6,      # 정규화 단위/초. 이보다 빠르면 FAST
    grab_drop: float = 0.45,       # openness가 한 프레임에 이만큼 줄면 GRAB
    score_thresh: float = 0.6,     # 이 점수 이상인 구간만 의심 구간으로
    merge_gap_sec: float = 0.4,    # 이 간격 이내 구간은 하나로 병합
) -> list[Segment]:
    weights = MODE_WEIGHTS.get(mode, MODE_WEIGHTS["card"])
    n = len(frames)
    if n == 0:
        return []

    # 손 개수 시퀀스(가장자리 손 제외한 '안정적으로 보이는' 손)
    counts = [sum(1 for h in f.hands if not h.at_border) for f in frames]
    rolling_max = 0
    per_frame: list[dict[str, float]] = []

    prev = None
    prev_open = {}  # label -> openness
    for i, f in enumerate(frames):
        sig = {s: 0.0 for s in SIGNALS}

        # VANISH: 직전 최대 손 개수보다 줄었을 때
        rolling_max = max(int(rolling_max * 0.9), counts[i])  # 천천히 감쇠
        if i > 0 and counts[i] < counts[i - 1]:
            sig["VANISH"] = float(counts[i - 1] - counts[i])

        # BORDER: 가장자리에 닿은 손이 있으면
        if any(h.at_border for h in f.hands):
            sig["BORDER"] = 1.0

        # CONTACT
        sig["CONTACT"] = _contact(f)

        # FAST
        v = _match_velocity(prev, f, fps)
        if v > fast_thresh:
            sig["FAST"] = min(1.5, v / fast_thresh)

        # GRAB: 같은 라벨 손의 openness 급감
        cur_open = {}
        for h in f.hands:
            cur_open[h.label] = h.openness
            if h.label in prev_open:
                drop = prev_open[h.label] - h.openness
                if drop > grab_drop:
                    sig["GRAB"] = max(sig["GRAB"], min(1.5, drop / grab_drop))
        prev_open = cur_open

        per_frame.append(sig)
        prev = f

    # 가중 점수 + 살짝 스무딩(앞뒤 평균)
    raw = np.array([sum(weights[s] * sig[s] for s in SIGNALS) for sig in per_frame])
    if n >= 3:
        smooth = np.convolve(raw, np.array([0.25, 0.5, 0.25]), mode="same")
    else:
        smooth = raw

    # 임계 초과 구간 추출
    segments: list[Segment] = []
    i = 0
    while i < n:
        if smooth[i] < score_thresh:
            i += 1
            continue
        j = i
        while j < n and smooth[j] >= score_thresh:
            j += 1
        seg_idx = range(i, j)
        peak = max(seg_idx, key=lambda k: smooth[k])
        agg: dict[str, float] = {s: 0.0 for s in SIGNALS}
        for k in seg_idx:
            for s in SIGNALS:
                agg[s] += weights[s] * per_frame[k][s]
        segments.append(Segment(
            start_sec=frames[i].time_sec,
            end_sec=frames[j - 1].time_sec,
            peak_sec=frames[peak].time_sec,
            score=float(smooth[peak]),
            signals={s: round(v, 3) for s, v in agg.items() if v > 0},
        ))
        i = j

    # 가까운 구간 병합
    merged: list[Segment] = []
    for seg in segments:
        if merged and seg.start_sec - merged[-1].end_sec <= merge_gap_sec:
            last = merged[-1]
            last.end_sec = seg.end_sec
            if seg.score > last.score:
                last.score = seg.score
                last.peak_sec = seg.peak_sec
            for s, v in seg.signals.items():
                last.signals[s] = round(last.signals.get(s, 0.0) + v, 3)
        else:
            merged.append(seg)

    merged.sort(key=lambda s: -s.score)
    return merged
