#!/usr/bin/env python
# coding: utf-8
"""
Send a Brevo email when PyEPO tmux experiments finish.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.error
import urllib.request


DEFAULT_ENV_PATH = Path.home() / ".config" / "experiment-notify" / "brevo.env"
DEFAULT_SESSIONS = [
    "pyepo_sp_lr",
    "pyepo_sp_rf",
    "pyepo_sp_spo",
    "pyepo_sp_dbb_pfyl_queue",
]
ERROR_PATTERNS = [
    "Traceback",
    "GurobiError",
    "FileNotFoundError",
    "No module named",
    "RuntimeError",
]


def load_env_file(path):
    env = {}
    path = Path(path)
    if not path.exists():
        return env
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        parts = shlex.split(value)
        env[key] = parts[0] if parts else ""
    return env


def tmux_session_exists(session):
    result = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def active_sessions(sessions):
    return [session for session in sessions if tmux_session_exists(session)]


def count_csv_rows(path):
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        rows = sum(1 for _ in reader)
    return max(rows - 1, 0)


def collect_summary(result_dir, log_dir):
    result_dir = Path(result_dir)
    log_dir = Path(log_dir)
    csv_files = sorted(result_dir.rglob("*.csv")) if result_dir.exists() else []
    log_files = sorted(log_dir.glob("*.log")) if log_dir.exists() else []
    error_logs = []
    for log_file in log_files:
        text = log_file.read_text(errors="replace")
        if any(pattern in text for pattern in ERROR_PATTERNS):
            error_logs.append(log_file.name)
    return {
        "csv_files": len(csv_files),
        "result_rows": sum(count_csv_rows(path) for path in csv_files),
        "log_files": len(log_files),
        "error_hits": len(error_logs),
        "error_logs": error_logs,
    }


def format_message(summary, sessions):
    error_text = "none" if not summary["error_logs"] else ", ".join(summary["error_logs"])
    return (
        "PyEPO SPO shortest-path experiments finished.\n\n"
        "Watched tmux sessions: {}\n"
        "Result CSV files: {}\n"
        "Result rows: {}\n"
        "Log files: {}\n"
        "Logs with error patterns: {}\n".format(
            ", ".join(sessions),
            summary["csv_files"],
            summary["result_rows"],
            summary["log_files"],
            error_text,
        )
    )


def send_brevo_email(api_key, sender, recipient, subject, text):
    payload = {
        "sender": {"email": sender},
        "to": [{"email": recipient}],
        "subject": subject,
        "textContent": text,
    }
    request = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "accept": "application/json",
            "api-key": api_key,
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("Brevo API failed with HTTP {}: {}".format(exc.code, body))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument("--result-dir", type=Path, default=Path("../res"))
    parser.add_argument("--log-dir", type=Path, default=Path("../logs"))
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--session", action="append", dest="sessions")
    parser.add_argument("--subject", default="PyEPO SPO experiments finished")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    sessions = args.sessions or DEFAULT_SESSIONS
    while True:
        running = active_sessions(sessions)
        if not running:
            break
        print("Still running: {}".format(", ".join(running)), flush=True)
        time.sleep(args.interval)

    summary = collect_summary(args.result_dir, args.log_dir)
    message = format_message(summary, sessions)
    if args.dry_run:
        print(message)
        return

    env = os.environ.copy()
    env.update(load_env_file(args.env))
    required = ["BREVO_API_KEY", "SPO_NOTIFY_FROM", "SPO_NOTIFY_TO"]
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise RuntimeError(
            "Missing notification settings in {}: {}".format(
                args.env,
                ", ".join(missing),
            )
        )
    status, body = send_brevo_email(
        env["BREVO_API_KEY"],
        env["SPO_NOTIFY_FROM"],
        env["SPO_NOTIFY_TO"],
        args.subject,
        message,
    )
    print("Brevo notification sent: HTTP {}".format(status))
    if body:
        print(body)


if __name__ == "__main__":
    main()
