"""명령행 진입점 — 영상 분석 → 리포트/마킹영상/의심프레임 출력."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from .detect import detect
from .fetch import resolve_video_input
from .hands import HandTracker
from .report import draw_overlay, format_report, to_dict, write_json
from .video import iter_frames, make_writer, open_video


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="magic-analyzer",
        description="카드/동전 마술 영상에서 '수상한 순간'을 찾아주는 분석 도구",
    )
    p.add_argument("video", help="분석할 영상 파일 경로 또는 YouTube URL")
    p.add_argument("--mode", choices=["card", "coin"], default="card",
                   help="마술 종류 (기본: card)")
    p.add_argument("--out", default="out", help="결과 출력 폴더 (기본: out)")
    p.add_argument("--stride", type=int, default=1,
                   help="N프레임마다 1장만 분석(속도용, 기본 1=전부)")
    p.add_argument("--annotate", action="store_true",
                   help="손 관절+의심구간이 표시된 영상 저장")
    p.add_argument("--save-frames", type=int, default=5, metavar="K",
                   help="상위 K개 의심 구간의 정점 프레임 이미지 저장 (기본 5, 0이면 끔)")
    p.add_argument("--score-thresh", type=float, default=0.9,
                   help="의심 봉우리 점수 임계값 (낮출수록 더 많이 잡음)")
    p.add_argument("--min-gap", type=float, default=1.2,
                   help="의심 순간 사이 최소 간격(초) — 가까운 봉우리는 하나로")
    p.add_argument("--window", type=float, default=0.8,
                   help="봉우리 앞뒤로 구간에 포함할 시간(초)")
    p.add_argument("--max-results", type=int, default=None,
                   help="최대 의심 순간 개수 (기본: 제한 없음)")
    p.add_argument("--llm", action="store_true",
                   help="의심 구간을 OpenAI 비전 모델로 추론 (OPENAI_API_KEY 환경변수 필요)")
    p.add_argument("--llm-model", default="gpt-4o",
                   help="OpenAI 비전 모델명 (기본 gpt-4o)")
    p.add_argument("--llm-top", type=int, default=5,
                   help="LLM으로 추론할 상위 구간 개수 (기본 5)")
    return p


def _load_env() -> None:
    """프로젝트 루트와 현재 디렉토리의 .env를 환경변수로 로드(OPENAI_API_KEY 등)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    for cand in (Path(__file__).resolve().parent.parent / ".env", Path.cwd() / ".env"):
        if cand.exists():
            load_dotenv(cand)


def _force_utf8() -> None:
    """Windows 콘솔(cp949)에서 한글/이모지 출력이 깨지거나 죽지 않도록 utf-8로 전환."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    _load_env()
    args = _build_parser().parse_args(argv)
    try:
        video_path = resolve_video_input(args.video)
    except Exception as e:
        print(f"[오류] 입력 처리 실패: {e}", file=sys.stderr)
        return 1
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
    segments = detect(frames, meta.fps, mode=args.mode,
                      score_thresh=args.score_thresh, min_gap_sec=args.min_gap,
                      window_sec=args.window, max_results=args.max_results)

    report_txt = format_report(segments, meta, args.mode)
    print()
    print(report_txt)

    (out_dir / "report.txt").write_text(report_txt, encoding="utf-8")
    write_json(out_dir / "report.json", to_dict(segments, meta, args.mode))

    # --- 3단계: 영상/프레임 출력 + (옵션) LLM용 프레임 수집 (영상 2차 패스) ---
    llm_segs = segments[:max(args.llm_top, 0)] if args.llm else []
    # LLM용: 각 구간의 직전/정점/직후 프레임 인덱스를 미리 계산
    off = max(1, int(0.4 * meta.fps))
    triplet_targets: dict[int, list[tuple[int, str]]] = {}
    triplet_store: dict[int, dict[str, "cv2.typing.MatLike"]] = {}
    for pos, s in enumerate(llm_segs):
        peak_idx = int(round(s.peak_sec * meta.fps))
        triplet_store[pos] = {}
        for slot, idx in (("before", peak_idx - off), ("peak", peak_idx),
                          ("after", peak_idx + off)):
            triplet_targets.setdefault(max(0, idx), []).append((pos, slot))

    need_second_pass = args.annotate or args.save_frames > 0 or bool(llm_segs)
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
            # LLM용 프레임 수집(오버레이 없는 원본)
            for pos, slot in triplet_targets.get(fr.index, ()):
                triplet_store[pos][slot] = fr.image.copy()
        if writer is not None:
            writer.release()
        cap.release()
        if args.annotate:
            print(f"      → {out_dir / 'annotated.mp4'}")
        if saved:
            print(f"      → 의심 프레임 {saved}장 저장")
    else:
        print("[3/3] (저장할 영상/프레임 없음)")

    # --- 4단계: LLM 추론 (옵션) ---
    if llm_segs:
        _run_llm(llm_segs, triplet_store, args, out_dir)

    print(f"\n완료. 결과 폴더: {out_dir.resolve()}")
    return 0


def _run_llm(llm_segs, triplet_store, args, out_dir: Path) -> None:
    """수집한 프레임을 OpenAI 비전 모델에 넘겨 구간별로 추론하고 결과를 저장."""
    from .llm import FrameTriplet, LLMInferencer

    print(f"[4/4] OpenAI({args.llm_model}) 추론 중... (상위 {len(llm_segs)}개 구간)")
    try:
        infer = LLMInferencer(model=args.llm_model)
    except Exception as e:  # 키 미설정/패키지 문제 등
        print(f"      [오류] OpenAI 초기화 실패: {e}", file=sys.stderr)
        print("      OPENAI_API_KEY 환경변수를 설정했는지 확인하세요.", file=sys.stderr)
        return

    lines = ["", "=" * 64, " LLM 추론 (OpenAI " + args.llm_model + ")", "=" * 64, ""]
    results = []
    for pos, seg in enumerate(llm_segs):
        store = triplet_store.get(pos, {})
        triplet = FrameTriplet(before=store.get("before"),
                               peak=store.get("peak"), after=store.get("after"))
        try:
            text = infer.infer(seg, triplet, args.mode)
        except Exception as e:
            text = f"(추론 실패: {e})"
        m, s = divmod(seg.peak_sec, 60)
        header = f"[{pos + 1}] {int(m):02d}:{s:05.2f} (점수 {seg.score:.2f})"
        print(f"      {header}\n        {text}")
        lines += [header, f"  {text}", ""]
        results.append({"rank": pos + 1, "peak_sec": round(seg.peak_sec, 2),
                        "inference": text})

    (out_dir / "llm.txt").write_text("\n".join(lines), encoding="utf-8")
    write_json(out_dir / "llm.json", {"model": args.llm_model, "results": results})
    print(f"      → {out_dir / 'llm.txt'}")


if __name__ == "__main__":
    raise SystemExit(main())
