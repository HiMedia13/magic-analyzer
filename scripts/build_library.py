"""기법 예시 라이브러리 구축 — 기법 시연의 손 궤적 시그니처를 모은다.

사용:
  # 자동: 기법 튜토리얼을 검색·다운로드해 '가장 의심스러운 순간'을 그 기법의 예시로 등록
  python scripts/build_library.py --auto "double lift" "french drop" "classic palm"

  # 수동: 영상/URL의 특정 시각을 기법 예시로 등록 (가장 정확)
  python scripts/build_library.py --technique "프렌치 드롭" --video coin.mp4 --time 12.3 --mode coin

자동 모드 가정: '기법 X 튜토리얼' 영상의 최상위 의심 순간 ≈ 기법 X의 한 사례.
거칠지만 데모 시드로는 충분. 정확도를 높이려면 수동으로 검증된 시각을 등록하세요.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from magic_analyzer.detect import detect  # noqa: E402
from magic_analyzer.fetch import download_video, is_url  # noqa: E402
from magic_analyzer.hands import HandTracker  # noqa: E402
from magic_analyzer.library import save_entry, signature_from_video  # noqa: E402
from magic_analyzer.techniques import lookup  # noqa: E402
from magic_analyzer.video import iter_frames, open_video  # noqa: E402


def _force_utf8():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _top_moment(video_path: str, mode: str) -> float | None:
    """영상에서 가장 점수 높은 의심 순간의 시각(초). 속도 위해 stride=2."""
    cap, meta = open_video(video_path)
    frames = []
    with HandTracker() as tr:
        for fr in iter_frames(cap, meta.fps, stride=2):
            frames.append(tr.process(fr.index, fr.time_sec, fr.image))
    cap.release()
    segs = detect(frames, meta.fps, mode=mode, max_results=1)
    return segs[0].peak_sec if segs else None


def _resolve(technique: str) -> tuple[str, str, str]:
    """기법명 → (key_en, name_ko, mode추정)."""
    e = lookup(technique)
    if e:
        mode = "coin" if "coin" in e["type"] else "card"
        return e["en"], e["ko"], mode
    return technique, technique, "card"


def _add(technique: str, video: str, time_sec: float, mode: str, source: str) -> bool:
    key, name_ko, _ = _resolve(technique)
    sig = signature_from_video(video, time_sec)
    if sig is None:
        print(f"  [실패] 시그니처 추출 불가(손 미검출): {technique} @ {time_sec:.1f}s")
        return False
    save_entry(key, name_ko, sig, source=source)
    print(f"  [등록] {key} ({name_ko}) @ {time_sec:.1f}s  ← {source}")
    return True


def _search_download(query: str) -> str | None:
    import yt_dlp
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
        r = ydl.extract_info(f"ytsearch1:{query} magic tutorial slow motion",
                             download=False)
        entries = r.get("entries") or []
        if not entries:
            return None
        url = entries[0].get("webpage_url") or entries[0].get("url")
    return str(download_video(url))


def main():
    _force_utf8()
    ap = argparse.ArgumentParser()
    ap.add_argument("--auto", nargs="+", metavar="TECH",
                    help="기법명들 — 각각 튜토리얼 검색·다운로드해 top 순간 등록")
    ap.add_argument("--technique", help="수동 등록할 기법명")
    ap.add_argument("--video", help="수동: 영상 경로/URL")
    ap.add_argument("--time", type=float, help="수동: 시각(초)")
    ap.add_argument("--mode", choices=["card", "coin"], default=None,
                    help="top 순간 탐지 모드(자동은 기법 종류로 추정)")
    args = ap.parse_args()

    if args.auto:
        for tech in args.auto:
            print(f"[자동] {tech}")
            _, name_ko, guess_mode = _resolve(tech)
            mode = args.mode or guess_mode
            try:
                vid = _search_download(f"{tech}")
            except Exception as e:
                print(f"  [실패] 다운로드: {e}")
                continue
            t = _top_moment(vid, mode)
            if t is None:
                print("  [실패] 의심 순간 없음")
                continue
            _add(tech, vid, t, mode, source=f"auto:{Path(vid).stem}@{t:.1f}")
        return 0

    if args.technique and args.video and args.time is not None:
        video = args.video if is_url(args.video) else str(Path(args.video))
        if is_url(video):
            video = str(download_video(video))
        _, _, guess = _resolve(args.technique)
        _add(args.technique, video, args.time, args.mode or guess,
             source=f"manual:{args.time:.1f}")
        return 0

    ap.error("--auto 또는 (--technique --video --time)을 지정하세요.")


if __name__ == "__main__":
    raise SystemExit(main())
