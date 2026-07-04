"""Transfermarkt rail: managed local server + market-value client.

TransfermarktServer bootstraps the felipeall/transfermarkt-api repo into
`vendor/`, creates a dedicated venv from its requirements.txt (no Poetry
needed), launches uvicorn as a background subprocess, health-checks it, and
stops it cleanly on exit. If a server is already listening on the port, it is
reused (and left running).

TransfermarktClient reads per-club market values, matched EXACTLY on club id.
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import niquests

from .config import Config
from .http_client import HttpClient
from .normalize import normalize_pid


class TransfermarktServer:
    def __init__(self, config: Config):
        self.config = config
        self.proc: subprocess.Popen | None = None
        self._started_by_us = False
        self._log_file = None

    # -- lifecycle --
    def __enter__(self) -> str:
        self.start()
        return self.config.tm_base_url

    def __exit__(self, *exc):
        self.stop()

    def start(self) -> str:
        """Ensure a Transfermarkt API is reachable; return its base URL."""
        if self.config.tm_base_url_override:
            print(f"Transfermarkt: using override {self.config.tm_base_url}")
            return self.config.tm_base_url

        if self._is_up():
            print(f"Transfermarkt: reusing server already at {self.config.tm_base_url}")
            return self.config.tm_base_url

        self._bootstrap_repo()
        venv_python = self._bootstrap_venv()
        self._launch(venv_python)
        self._wait_until_up()
        atexit.register(self.stop)  # safety net if __exit__ is skipped
        return self.config.tm_base_url

    def stop(self) -> None:
        if self.proc is None or not self._started_by_us:
            return
        print("Transfermarkt: stopping local server")
        try:
            # kill the whole process group so uvicorn workers can't be orphaned
            try:
                pgid = os.getpgid(self.proc.pid)
                os.killpg(pgid, signal.SIGTERM)
            except Exception:
                self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    self.proc.kill()
        except Exception:
            pass
        finally:
            if self._log_file:
                self._log_file.close()
            self.proc = None
            self._started_by_us = False

    # -- health --
    def _is_up(self) -> bool:
        try:
            r = niquests.get(f"{self.config.tm_base_url}/players/search/Messi", timeout=5)
            return r.status_code < 500
        except Exception:
            return False

    def _wait_until_up(self, timeout: int = 60) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(
                    f"Transfermarkt server exited early (code {self.proc.returncode}); "
                    f"see log: {self._log_path()}"
                )
            if self._is_up():
                print(f"Transfermarkt: server ready at {self.config.tm_base_url}")
                return
            time.sleep(1.0)
        raise TimeoutError(f"Transfermarkt server did not become ready within {timeout}s")

    # -- bootstrap --
    def _bootstrap_repo(self) -> None:
        repo = self.config.tm_repo_dir
        if not (repo / "app" / "main.py").exists():
            self.config.vendor_dir.mkdir(parents=True, exist_ok=True)
            print(f"Transfermarkt: cloning repo into {repo}")
            subprocess.run(
                ["git", "clone", self.config.tm_repo_url, str(repo)],
                check=True,
            )
        self._checkout_pinned_commit(repo)

    def _checkout_pinned_commit(self, repo: Path) -> None:
        """Pin the vendored repo to the commit declared in Config (reproducibility)."""
        commit = self.config.tm_repo_commit
        if not commit:
            return
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        if head == commit:
            return
        print(f"Transfermarkt: checking out pinned commit {commit[:12]}")
        # A pre-existing shallow clone may not contain the pinned commit yet.
        if subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
        ).returncode != 0:
            subprocess.run(
                ["git", "-C", str(repo), "fetch", "origin", commit],
                check=True,
            )
        subprocess.run(
            ["git", "-C", str(repo), "checkout", "--detach", commit],
            check=True,
        )

    def _bootstrap_venv(self) -> Path:
        repo = self.config.tm_repo_dir
        venv = repo / ".venv"
        venv_python = venv / "bin" / "python"
        marker = venv / ".deps_installed_v2"
        if venv_python.exists() and marker.exists():
            return venv_python

        print(f"Transfermarkt: creating venv ({self.config.tm_python}) + installing deps")
        if not venv_python.exists():
            subprocess.run([self.config.tm_python, "-m", "venv", str(venv)], check=True)
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-q", "--upgrade", "pip"], check=True
        )
        deps = ["-r", str(repo / "requirements.txt")]
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-q", *deps], check=True
        )
        marker.write_text("ok", encoding="utf-8")
        return venv_python

    def _log_path(self) -> Path:
        return self.config.tm_repo_dir / "server.log"

    def _launch(self, venv_python: Path) -> None:
        repo = self.config.tm_repo_dir
        c = self.config
        self._log_file = open(self._log_path(), "w", encoding="utf-8")
        env = {**_os_environ(), "RATE_LIMITING_ENABLE": "false", "PYTHONPATH": str(repo)}
        print(f"Transfermarkt: launching uvicorn on :{c.tm_port} "
              f"({c.tm_server_workers} workers)")
        self.proc = subprocess.Popen(
            [
                str(venv_python), "-m", "uvicorn", "app.main:app",
                "--host", c.tm_host, "--port", str(c.tm_port),
                "--workers", str(c.tm_server_workers),
            ],
            cwd=str(repo),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # own process group -> clean group kill
        )
        self._started_by_us = True


def _os_environ() -> dict:
    import os
    return dict(os.environ)


class TransfermarktClient:
    def __init__(self, http: HttpClient):
        self.http = http

    def history_map(self, transfermarkt_player_id) -> dict:
        """GET /players/{id}/market_value -> {club_id: marketValue}."""
        pid = normalize_pid(transfermarkt_player_id)
        if not pid:
            return {}
        data = self.http.get_tm_json(
            f"/players/{pid}/market_value", f"{pid}__market_value.json"
        )
        out = {}
        for entry in data.get("marketValueHistory") or []:
            cid = normalize_pid(entry.get("clubId"))
            if cid:
                out[cid] = entry.get("marketValue")  # chronological -> latest per club
        return out

    def transfers_map(self, transfermarkt_player_id) -> dict:
        """GET /players/{id}/transfers -> {clubTo_id: marketValue} (latest arrival)."""
        pid = normalize_pid(transfermarkt_player_id)
        if not pid:
            return {}
        data = self.http.get_tm_json(
            f"/players/{pid}/transfers", f"{pid}__transfers.json"
        )
        out = {}
        for t in sorted(data.get("transfers") or [], key=lambda x: x.get("date") or ""):
            cid = normalize_pid((t.get("clubTo") or {}).get("id"))
            mv = t.get("marketValue")
            if cid and mv is not None:
                out[cid] = mv
        return out

    @staticmethod
    def resolve(team_id, history_map: dict, transfers_map: dict):
        """Market value for a club id: history -> transfers -> None (no fallback)."""
        key = normalize_pid(team_id)
        if not key:
            return None
        if key in history_map:
            return history_map[key]
        if key in transfers_map:
            return transfers_map[key]
        return None
