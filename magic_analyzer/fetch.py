"""입력 해석 — 로컬 경로면 그대로, URL이면 yt-dlp로 받아 로컬 경로를 돌려준다.

해상도는 분석 정확도에 거의 영향이 없으므로(README 팁 참고) 360p progressive(mp4
단일 스트림, ffmpeg 불필요)를 받는다. 같은 영상은 id로 캐시해 재다운로드를 피한다.
"""

from __future__ import annotations

from pathlib import Path

# 받은 영상 캐시 위치(프로젝트 루트/downloads, git 제외)
DOWNLOAD_DIR = Path(__file__).resolve().parent.parent / "downloads"

# 360p progressive(18) 우선, 없으면 480p 이하 단일 mp4 → 마지막으로 아무거나
_FORMAT = "18/b[ext=mp4][height<=480]/b[ext=mp4]/b"


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def download_video(url: str) -> Path:
    """URL을 받아 로컬 mp4 경로를 반환. yt-dlp 필요."""
    try:
        import yt_dlp
    except ImportError as e:
        raise RuntimeError(
            "URL 분석에는 yt-dlp가 필요합니다: pip install yt-dlp"
        ) from e

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": _FORMAT,
        "outtmpl": str(DOWNLOAD_DIR / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        # 1) 메타데이터만 먼저 조회해 영상 id를 얻는다.
        meta = ydl.extract_info(url, download=False)
        vid = meta.get("id")
        # 2) 이미 받아둔 완성 파일이 있으면 재다운로드 없이 그대로 사용
        #    (멀티모드에서 같은 URL을 두 번 받으며 .part 이름변경 충돌 나는 것 방지)
        if vid:
            for ext in ("mp4", "webm", "mkv", "m4a"):
                cand = DOWNLOAD_DIR / f"{vid}.{ext}"
                if cand.exists() and cand.stat().st_size > 0:
                    return cand
            # 끊겨 남은 .part 잔여물 정리(이름변경 충돌 예방)
            for stale in DOWNLOAD_DIR.glob(f"{vid}.*.part"):
                try:
                    stale.unlink()
                except OSError:
                    pass
        # 3) 실제 다운로드
        info = ydl.extract_info(url, download=True)
        reqs = info.get("requested_downloads") or []
        if reqs and reqs[0].get("filepath"):
            return Path(reqs[0]["filepath"])
        return Path(ydl.prepare_filename(info))


def resolve_video_input(video: str) -> Path:
    """CLI 인자를 로컬 Path로 해석. URL이면 다운로드해서 그 경로를 반환."""
    if is_url(video):
        print(f"      URL 감지 → 다운로드 중... ({video})")
        path = download_video(video)
        print(f"      → {path.name}")
        return path
    return Path(video)
