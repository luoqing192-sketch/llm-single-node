"""隔离代码沙箱：在受限子进程里执行短 Python 片段。"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class ExecRequest(BaseModel):
    code: str = Field(min_length=1, max_length=20000)
    timeout_sec: float = Field(default=5, ge=0.5, le=30)


class ExecResponse(BaseModel):
    ok: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_ms: int = 0


app = FastAPI(title="Single Node LLM Sandbox")


def _posix_limits(timeout_sec: float):
    def apply():
        try:
            import resource
            cpu = max(1, int(timeout_sec) + 1)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            mem = 256 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
        except (ImportError, ValueError, OSError):
            pass
    return apply


def run_isolated(code: str, timeout_sec: float) -> ExecResponse:
    work = Path(tempfile.mkdtemp(prefix="sandbox-"))
    started = time.time()
    try:
        script = work / "main.py"
        script.write_text(code, encoding="utf-8")
        env = {
            "HOME": str(work),
            "TMPDIR": str(work),
            "PYTHONIOENCODING": "utf-8",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
        }
        kwargs = {
            "args": [sys.executable, "-I", str(script)],
            "cwd": str(work),
            "capture_output": True,
            "text": True,
            "timeout": timeout_sec,
            "env": env,
        }
        if os.name == "posix":
            kwargs["preexec_fn"] = _posix_limits(timeout_sec)
        try:
            completed = subprocess.run(**kwargs)
        except subprocess.TimeoutExpired as exc:
            return ExecResponse(
                ok=False,
                stdout=(exc.stdout or "")[-4000:],
                stderr=((exc.stderr or "") + "\ntimeout")[-4000:],
                timed_out=True,
                duration_ms=int((time.time() - started) * 1000),
            )
        stdout = (completed.stdout or "")[-8000:]
        stderr = (completed.stderr or "")[-8000:]
        return ExecResponse(
            ok=completed.returncode == 0,
            exit_code=completed.returncode,
            stdout=stdout,
            stderr=stderr,
            duration_ms=int((time.time() - started) * 1000),
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.get("/health")
def health():
    return {"status": "ok", "isolation": "subprocess+rlimit"}


@app.post("/exec", response_model=ExecResponse)
def exec_code(request: ExecRequest):
    if "\x00" in request.code:
        raise HTTPException(status_code=400, detail="invalid code")
    return run_isolated(request.code, request.timeout_sec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
