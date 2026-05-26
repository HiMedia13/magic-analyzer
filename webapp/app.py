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
_jobs: dict[str, dict] = {}  # job_id -> {proc, dir}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    mode = request.form.get("mode", "card")
    if mode not in ("card", "coin"):
        mode = "card"

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

    cmd = [PYTHON, str(MAIN), video_arg, "--mode", mode,
           "--out", str(job_dir), "--save-frames", "6", "--max-results", "12"]
    if request.form.get("annotate") == "on":
        cmd.append("--annotate")
    if request.form.get("llm") == "on":
        cmd += ["--llm", "--llm-top", "6"]

    log = open(job_dir / "log.txt", "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            cwd=str(PROJECT))
    _jobs[job_id] = {"proc": proc, "dir": job_dir}
    return jsonify({"job_id": job_id})


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@app.route("/status/<job_id>")
def status(job_id):
    job = _jobs.get(job_id)
    if not job:
        abort(404)
    proc, job_dir = job["proc"], job["dir"]
    rc = proc.poll()
    if rc is None:
        return jsonify({"status": "running"})

    log_tail = ""
    log_path = job_dir / "log.txt"
    if log_path.exists():
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]

    if rc != 0:
        return jsonify({"status": "failed", "log": log_tail})

    report = _read_json(job_dir / "report.json") or {}
    llm = _read_json(job_dir / "llm.json")
    # LLM 결과를 peak_sec로 매칭해 세그먼트에 붙임
    llm_by_peak = {}
    if llm:
        for r in llm.get("results", []):
            llm_by_peak[round(r.get("peak_sec", -1), 1)] = r.get("inference", "")
    for seg in report.get("segments", []):
        key = round(seg.get("peak_sec", -2), 1)
        if key in llm_by_peak:
            seg["inference"] = llm_by_peak[key]

    frames = sorted(p.name for p in job_dir.glob("suspect_*.jpg"))
    return jsonify({
        "status": "done",
        "job_id": job_id,
        "report": report,
        "summary": (llm or {}).get("summary"),  # 에이전트 전체 트릭 추정
        "techniques": (llm or {}).get("techniques", []),  # 기법 설명 + 참고 영상
        "frames": frames,
        "video": (job_dir / "annotated.webm").exists(),
        "log": log_tail,
    })


@app.route("/jobs/<job_id>/<path:filename>")
def job_file(job_id, filename):
    # 경로 탈출 방지: secure_filename으로 제한
    safe = secure_filename(filename)
    if not safe:
        abort(404)
    return send_from_directory(JOBS_DIR / job_id, safe)


if __name__ == "__main__":
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
