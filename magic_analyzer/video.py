"""영상 입출력 유틸 — 프레임을 타임스탬프와 함께 읽고, 분석 영상을 쓴다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int

    @property
    def duration_sec(self) -> float:
        return self.frame_count / self.fps if self.fps else 0.0


@dataclass
class Frame:
    index: int
    time_sec: float
    image: "cv2.typing.MatLike"  # BGR


def open_video(path: str | Path) -> tuple[cv2.VideoCapture, VideoMeta]:
    path = str(path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    meta = VideoMeta(
        fps=fps,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    return cap, meta


def iter_frames(cap: cv2.VideoCapture, fps: float, stride: int = 1) -> Iterator[Frame]:
    """프레임을 순서대로 내보낸다. stride>1이면 N프레임마다 1장만 처리(속도용)."""
    idx = 0
    while True:
        ok, image = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            yield Frame(index=idx, time_sec=idx / fps, image=image)
        idx += 1


def make_writer(path: str | Path, meta: VideoMeta) -> cv2.VideoWriter:
    # .webm은 VP8(브라우저 재생 가능), 그 외는 mp4v. 이 OpenCV 빌드는 H.264 인코더가
    # 없어 avc1가 안 되므로, 웹 표시는 webm을 쓴다.
    tag = "VP80" if Path(path).suffix.lower() == ".webm" else "mp4v"
    fourcc = cv2.VideoWriter_fourcc(*tag)
    writer = cv2.VideoWriter(str(path), fourcc, meta.fps, (meta.width, meta.height))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter 초기화 실패(코덱 {tag} 미지원?): {path}")
    return writer
