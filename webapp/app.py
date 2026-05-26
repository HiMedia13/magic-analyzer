"""magic-analyzer 웹 UI (Flask).

브라우저에서 YouTube URL/파일 업로드 → 모드·옵션 선택 → 분석 결과(타임라인,
의심 프레임, 마킹 영상, AI 설명)를 한 화면에 보여준다.

분석은 오래 걸리므로 검증된 CLI(main.py)를 백그라운드 서브프로세스로 돌리고,
브라우저가 /status/<job>을 폴링한다. 결과물은 job 디렉토리의 파일을 그대로 읽는다.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from flask import (Flask, abort, jsonify, render_template, request,
                   send_from_directory)
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
JOBS_DIR = BASE / "jobs"
MAIN = PROJECT / "main.py"
PYTHON = sys.executable  # 이 앱을 돌리는 venv 파이썬 그대로 사용

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 업로드 최대 500MB
_jobs: dict[str, dict] = {}  # job_id -> {proc, dir, modes, idx, video_arg, ...}
_status_lock = threading.Lock()  # /status 시퀀서 전이를 원자화(중복 spawn 방지)

MODES = ("card", "coin")  # 분석 가능한 마술 종류


@app.route("/")
def index():
    return render_template("index.html")


def _spawn(job: dict, mode: str) -> subprocess.Popen:
    """선택된 모드 하나에 대해 CLI(main.py)를 jobs/<id>/<mode>/ 로 출력하며 실행."""
    out_dir = job["dir"] / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(MAIN), job["video_arg"], "--mode", mode,
           "--out", str(out_dir), "--save-frames", "6", "--max-results", "12"]
    if job["annotate"]:
        cmd.append("--annotate")
    if job["llm"]:
        cmd += ["--llm", "--llm-top", "6"]
    log = open(out_dir / "log.txt", "w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            cwd=str(PROJECT))


@app.route("/analyze", methods=["POST"])
def analyze():
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # 모드 다중 선택: 체크된 것 중 유효한 것만, 순서 유지·중복 제거. 없으면 card.
    modes = [m for m in request.form.getlist("mode") if m in MODES]
    modes = list(dict.fromkeys(modes)) or ["card"]

    url = (request.form.get("url") or "").strip()
    if url:
        video_arg = url
    else:
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"error": "YouTube URL을 입력하거나 영상 파일을 올려주세요."}), 400
        dest = job_dir / ("input_" + secure_filename(f.filename))
        f.save(dest)
        video_arg = str(dest)

    job = {
        "dir": job_dir, "modes": modes, "idx": 0, "video_arg": video_arg,
        "annotate": request.form.get("annotate") == "on",
        "llm": request.form.get("llm") == "on",
    }
    # 모드는 순차 실행(같은 URL 동시 다운로드 충돌·CV 리소스 경쟁 방지).
    # /status가 한 모드 완료를 감지하면 다음 모드를 띄우는 시퀀서 역할을 한다.
    job["proc"] = _spawn(job, modes[0])
    _jobs[job_id] = job
    return jsonify({"job_id": job_id})


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect(mode_dir: Path, mode: str) -> dict:
    """한 모드 폴더의 결과(리포트·LLM·프레임·마킹영상)를 한 묶음으로 모은다."""
    report = _read_json(mode_dir / "report.json") or {}
    llm = _read_json(mode_dir / "llm.json")
    # LLM 결과를 peak_sec로 매칭해 세그먼트에 붙임
    llm_by_peak = {}
    if llm:
        for r in llm.get("results", []):
            llm_by_peak[round(r.get("peak_sec", -1), 1)] = r.get("inference", "")
    for seg in report.get("segments", []):
        key = round(seg.get("peak_sec", -2), 1)
        if key in llm_by_peak:
            seg["inference"] = llm_by_peak[key]
    frames = sorted(p.name for p in mode_dir.glob("suspect_*.jpg"))
    return {
        "mode": mode,
        "report": report,
        "summary": (llm or {}).get("summary"),     # 에이전트 전체 트릭 추정
        "techniques": (llm or {}).get("techniques", []),  # 기법 설명 + 참고 영상
        "frames": frames,
        "video": (mode_dir / "annotated.webm").exists(),
    }


@app.route("/status/<job_id>")
def status(job_id):
    job = _jobs.get(job_id)
    if not job:
        abort(404)

    # 전이(다음 모드 spawn)는 원자적이어야 한다 — 동시 폴링이 같은 전이를 두 번
    # 수행해 같은 URL을 동시 다운로드하면 yt-dlp 'unable to rename file'이 난다.
    with _status_lock:
        modes, idx = job["modes"], job["idx"]
        cur_mode = modes[idx]
        rc = job["proc"].poll()

        log_tail = ""
        log_path = job["dir"] / cur_mode / "log.txt"
        if log_path.exists():
            log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]

        progress = f"{idx + 1}/{len(modes)}"

        if rc is None:  # 현재 모드 진행 중
            return jsonify({"status": "running", "mode": cur_mode,
                            "progress": progress, "log": log_tail})

        if rc != 0:  # 현재 모드 실패
            return jsonify({"status": "failed", "mode": cur_mode,
                            "progress": progress, "log": log_tail})

        # 현재 모드 완료 → 다음 모드가 남았으면 띄우고 계속 running
        if idx + 1 < len(modes):
            job["idx"] = idx + 1
            job["proc"] = _spawn(job, modes[idx + 1])
            return jsonify({"status": "running", "mode": modes[idx + 1],
                            "progress": f"{idx + 2}/{len(modes)}", "log": log_tail})

        # 모든 모드 완료 → 모드별 결과 묶음 반환
        results = [_collect(job["dir"] / m, m) for m in modes]
        return jsonify({"status": "done", "job_id": job_id, "results": results,
                        "log": log_tail})


@app.route("/jobs/<job_id>/<mode>/<path:filename>")
def job_file(job_id, mode, filename):
    # 모드 화이트리스트 + secure_filename으로 경로 탈출 방지
    if mode not in MODES:
        abort(404)
    safe = secure_filename(filename)
    if not safe:
        abort(404)
    return send_from_directory(JOBS_DIR / job_id / mode, safe)


if __name__ == "__main__":
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
