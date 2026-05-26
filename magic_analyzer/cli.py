"""명령행 진입점 — 영상 분석 → 리포트/마킹영상/의심프레임 출력."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .detect import detect
from .hands import HandTracker
from .report import draw_overlay, format_report, to_dict, write_json
from .video import iter_frames, make_writer, open_video


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="magic-analyzer",
        description="카드/동전 마술 영상에서 '수상한 순간'을 찾아주는 분석 도구",
    )
    p.add_argument("video", help="분석할 영상 파일 경로 (mp4/mov 등)")
    p.add_argument("--mode", choices=["card", "coin"], default="card",
                   help="마술 종류 (기본: card)")
    p.add_argument("--out", default="out", help="결과 출력 폴더 (기본: out)")
    p.add_argument("--stride", type=int, default=1,
                   help="N프레임마다 1장만 분석(속도용, 기본 1=전부)")
    p.add_argument("--annotate", action="store_true",
                   help="손 관절+의심구간이 표시된 영상 저장")
    p.add_argument("--save-frames", type=int, default=5, metavar="K",
                   help="상위 K개 의심 구간의 정점 프레임 이미지 저장 (기본 5, 0이면 끔)")
    p.add_argument("--score-thresh", type=float, default=0.6,
                   help="의심 구간 점수 임계값 (낮출수록 더 많이 잡음)")
    return p


def _force_utf8() -> None:
    """Windows 콘솔(cp949)에서 한글/이모지 출력이 깨지거나 죽지 않도록 utf-8로 전환."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    args = _build_parser().parse_args(argv)
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[오류] 파일이 없습니다: {video_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1단계: 손 추적 ---
    print(f"[1/3] 손 추적 중... ({video_path.name}, mode={args.mode})")
    cap, meta = open_video(video_path)
    frames = []
    with HandTracker() as tracker:
        for fr in iter_frames(cap, meta.fps, stride=args.stride):
            frames.append(tracker.process(fr.index, fr.time_sec, fr.image))
    cap.release()
    print(f"      처리 프레임: {len(frames)}장, fps={meta.fps:.1f}, "
          f"길이={meta.duration_sec:.1f}s")

    # --- 2단계: 의심 구간 탐지 ---
    print("[2/3] 의심 구간 분석 중...")
    segments = detect(frames, meta.fps, mode=args.mode, score_thresh=args.score_thresh)

    report_txt = format_report(segments, meta, args.mode)
    print()
    print(report_txt)

    (out_dir / "report.txt").write_text(report_txt, encoding="utf-8")
    write_json(out_dir / "report.json", to_dict(segments, meta, args.mode))

    # --- 3단계: 영상/프레임 출력 (필요 시 영상 2차 패스) ---
    need_second_pass = args.annotate or args.save_frames > 0
    if need_second_pass and segments:
        print("[3/3] 마킹 영상/프레임 저장 중...")
        obs_by_index = {f.index: f for f in frames}
        peaks = {round(s.peak_sec, 2): s for s in segments[:max(args.save_frames, 0)]}
        cap, meta = open_video(video_path)
        writer = make_writer(out_dir / "annotated.mp4", meta) if args.annotate else None
        saved = 0
        for fr in iter_frames(cap, meta.fps, stride=1):
            obs = obs_by_index.get(fr.index)
            active = next((s for s in segments
                           if s.start_sec <= fr.time_sec <= s.end_sec), None)
            if writer is not None:
                writer.write(draw_overlay(fr.image, obs, active))
            # 정점 프레임 저장
            if args.save_frames > 0:
                key = round(fr.time_sec, 2)
                if key in peaks:
                    s = peaks.pop(key)
                    fn = out_dir / f"suspect_{saved + 1:02d}_{key:.2f}s.jpg"
                    cv2.imwrite(str(fn), draw_overlay(fr.image, obs, s))
                    saved += 1
        if writer is not None:
            writer.release()
        cap.release()
        if args.annotate:
            print(f"      → {out_dir / 'annotated.mp4'}")
        if saved:
            print(f"      → 의심 프레임 {saved}장 저장")
    else:
        print("[3/3] (저장할 영상/프레임 없음)")

    print(f"\n완료. 결과 폴더: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
