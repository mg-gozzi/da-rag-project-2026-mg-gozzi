import logging
import subprocess
import sys
from pathlib import Path


LOG_PATH = Path(__file__).resolve().parent / "server.log"


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.info("Starting uvicorn server with reload enabled")
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--reload",
        "--log-level",
        "info",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        cwd=Path(__file__).resolve().parent,
    )

    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        while True:
            line = process.stdout.readline()
            if line == "" and process.poll() is not None:
                break
            if line:
                log_file.write(line)
                log_file.flush()
                sys.stdout.write(line)
                sys.stdout.flush()

    return_code = process.wait()
    logging.info(f"Server process exited with code {return_code}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
