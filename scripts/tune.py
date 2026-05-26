"""개발용 임계값 튜닝 스크립트.

손 추적 결과(FrameObs 리스트)를 pickle로 캐시한 뒤, detect 파라미터만 바꿔가며
빠르게 결과를 비교한다. 추적은 영상당 한 번만 수행한다.

사용:
    python scripts/tune.py samples/coin.mp4 --mode coin \
        --score-thresh 1.2 --min-gap 1.5 --window 0.8
    python scripts/tune.py samples/coin.mp4 --mode coin --hist   # 점수 분포만
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from magic_analyzer.detect import detect, score_timeline  # noqa: E402
from magic_analyzer.hands import HandTracker  # noqa: E402
from magic_analyzer.video import iter_frames, open_video  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent.parent / "out" / "_cache"


def _fmt(t: float) -> str:
    m, s = divmod(t, 60)
    return f"{int(m):02d}:{s:05.2f}"


def load_frames(video: Path):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / (video.stem + ".pkl")
    if cache.exists():
        with open(cache, "rb") as f:
            data = pickle.load(f)
        return data["frames"], data["fps"]

    print(f"[추적] {video.name} (캐시 없음, 최초 1회)...")
    cap, meta = open_video(video)
    frames = []
    with HandTracker() as tracker:
        for fr in iter_frames(cap, meta.fps, stride=1):
            frames.append(tracker.process(fr.index, fr.time_sec, fr.image))
    cap.release()
    with open(cache, "wb") as f:
        pickle.dump({"frames": frames, "fps": meta.fps}, f)
    print(f"[추적] 완료 → 캐시 저장 {cache.name}")
    return frames, meta.fps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--mode", choices=["card", "coin"], default="card")
    ap.add_argument("--score-thresh", type=float, default=0.9)
    ap.add_argument("--min-gap", type=float, default=1.2)
    ap.add_argument("--window", type=float, default=0.8)
    ap.add_argument("--fast-thresh", type=float, default=1.6)
    ap.add_argument("--grab-drop", type=float, default=0.45)
    ap.add_argument("--border-margin", type=float, default=0.06)
    ap.add_argument("--max-results", type=int, default=12)
    ap.add_argument("--hist", action="store_true", help="점수 분포만 출력")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    frames, fps = load_frames(Path(args.video))

    _, smooth, _ = score_timeline(
        frames, fps, args.mode,
        fast_thresh=args.fast_thresh, grab_drop=args.grab_drop,
        border_margin=args.border_margin,
    )
    pct = [50, 75, 90, 95, 99]
    qs = np.percentile(smooth, pct)
    print(f"\n점수 분포 (mode={args.mode}, n={len(smooth)}): "
          f"max={smooth.max():.2f}, mean={smooth.mean():.2f}")
    print("  " + "  ".join(f"p{p}={q:.2f}" for p, q in zip(pct, qs)))
    if args.hist:
        return 0

    segs = detect(
        frames, fps, args.mode,
        fast_thresh=args.fast_thresh, grab_drop=args.grab_drop,
        border_margin=args.border_margin, score_thresh=args.score_thresh,
        min_gap_sec=args.min_gap, window_sec=args.window, max_results=args.max_results,
    )
    print(f"\n채택된 의심 순간: {len(segs)}개  "
          f"(thresh={args.score_thresh}, gap={args.min_gap}s, win={args.window}s)")
    for i, s in enumerate(segs):
        sigs = " ".join(f"{k}:{v}" for k, v in
                        sorted(s.signals.items(), key=lambda kv: -kv[1]))
        dur = s.end_sec - s.start_sec
        print(f"  [{i + 1:2d}] peak {_fmt(s.peak_sec)}  score={s.score:.2f}  "
              f"({dur:.1f}s)  {sigs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
