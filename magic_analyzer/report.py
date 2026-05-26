"""결과 출력 — 텍스트/JSON 리포트, 손 관절 오버레이, 의심 구간 배너 그리기."""

from __future__ import annotations

import json

import cv2
import numpy as np

from .detect import SIGNAL_DESC, Segment
from .hands import FrameObs
from .video import VideoMeta


def _fmt_time(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def to_dict(segments: list[Segment], meta: VideoMeta, mode: str) -> dict:
    return {
        "mode": mode,
        "duration_sec": round(meta.duration_sec, 2),
        "suspect_count": len(segments),
        "segments": [
            {
                "rank": i + 1,
                "start_sec": round(s.start_sec, 2),
                "end_sec": round(s.end_sec, 2),
                "peak_sec": round(s.peak_sec, 2),
                "score": round(s.score, 3),
                "signals": s.signals,
                "top_signals": s.top_signals,
            }
            for i, s in enumerate(segments)
        ],
    }


def format_report(segments: list[Segment], meta: VideoMeta, mode: str, top: int = 10) -> str:
    lines = [
        "=" * 64,
        f" 마술 영상 분석 리포트  (모드: {mode})",
        f" 길이: {_fmt_time(meta.duration_sec)}  |  의심 구간: {len(segments)}개",
        "=" * 64,
        "",
        "[주의] 이 결과는 '확정된 비밀'이 아니라 통계적으로 수상한 구간입니다.",
        "    아래 구간을 직접 돌려보며 무슨 일이 있었는지 확인하세요.",
        "",
    ]
    if not segments:
        lines.append("의심할 만한 구간을 찾지 못했습니다. (임계값을 낮춰 다시 시도해 보세요)")
        return "\n".join(lines)

    for i, s in enumerate(segments[:top]):
        lines.append(f"[{i + 1}] {_fmt_time(s.start_sec)} ~ {_fmt_time(s.end_sec)}"
                     f"   (정점 {_fmt_time(s.peak_sec)}, 점수 {s.score:.2f})")
        for sig in s.top_signals:
            lines.append(f"      · {sig}: {SIGNAL_DESC[sig]}")
        lines.append("")
    if len(segments) > top:
        lines.append(f"... 외 {len(segments) - top}개 구간 (전체는 JSON 리포트 참고)")
    return "\n".join(lines)


def draw_overlay(image, obs: FrameObs | None, active: Segment | None) -> "np.ndarray":
    """손 관절을 그리고, 의심 구간이면 상단에 빨간 배너를 표시한다."""
    out = image.copy()
    if obs is not None:
        for h in obs.hands:
            # 정규화 좌표 → 픽셀 좌표로 환산해 점/박스 그리기
            h_, w_ = out.shape[:2]
            pts = (h.landmarks * np.array([w_, h_])).astype(int)
            for (x, y) in pts:
                cv2.circle(out, (x, y), 3, (0, 255, 0), -1)
            xmin, ymin, xmax, ymax = h.bbox
            cv2.rectangle(out, (int(xmin * w_), int(ymin * h_)),
                          (int(xmax * w_), int(ymax * h_)), (0, 200, 0), 1)

    if active is not None:
        h_, w_ = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w_, 38), (0, 0, 200), -1)
        label = "SUSPECT  " + " / ".join(active.top_signals[:3])
        cv2.putText(out, label, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_json(path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
