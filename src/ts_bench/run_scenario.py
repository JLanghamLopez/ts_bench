import argparse
import asyncio
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import httpx
import psutil
import tomli as tomllib
from a2a.client import A2ACardResolver
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv(override=True)


def kill_process_tree(proc: subprocess.Popen, timeout: float = 5) -> None:
    """Kill a process and all its children cross-platform using psutil."""
    if proc.poll() is not None:
        return

    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)

        # Terminate children first, then parent
        for child in children:
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass

        parent.terminate()

        # Wait for graceful termination
        gone, alive = psutil.wait_procs([parent] + children, timeout=timeout)

        # Force kill any remaining processes
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass

    except psutil.NoSuchProcess:
        pass


async def wait_for_agents(cfg: dict, timeout: int = 30) -> bool:
    """Wait for all agents to be healthy and responding."""
    endpoints = []

    if cfg["participants"][0].get("cmd"):
        endpoints.append(
            f"http://{cfg['participants'][0]['host']}:{cfg['participants'][0]['port']}"
        )

    if cfg["green_agent"].get("cmd"):
        endpoints.append(
            f"http://{cfg['green_agent']['host']}:{cfg['green_agent']['port']}"
        )

    if not endpoints:
        return True  # No agents to wait for

    logger.info(f"Waiting for {len(endpoints)} agent(s) to be ready...")
    start_time = time.time()

    async def check_endpoint(_endpoint: str) -> bool:
        """Check if an endpoint is responding by fetching the agent card."""
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resolver = A2ACardResolver(httpx_client=client, base_url=_endpoint)
                await resolver.get_agent_card()
                return True
        except Exception as e:
            # Any exception means the agent is not ready
            logger.error(f"{_endpoint} : {e}")
            return False

    ready_count = 0

    while time.time() - start_time < timeout:
        ready_count = 0
        for endpoint in endpoints:
            if await check_endpoint(endpoint):
                ready_count += 1

        if ready_count == len(endpoints):
            return True

        logger.info(f"  {ready_count}/{len(endpoints)} agents ready, waiting...")
        await asyncio.sleep(1)

    logger.info(
        (
            f"Timeout: Only {ready_count}/{len(endpoints)} agents "
            f"became ready after {timeout}s"
        )
    )
    return False


def parse_toml(scenario_path: str) -> dict:
    path = Path(scenario_path)
    if not path.exists():
        print(f"Error: Scenario file not found: {path}")
        sys.exit(1)

    data = tomllib.loads(path.read_text())

    def host_port(ep: str):
        s = ep or ""
        s = s.replace("http://", "").replace("https://", "")
        s = s.split("/", 1)[0]
        host, port = s.split(":", 1)
        return host, int(port)

    green_ep = data.get("green_agent", {}).get("endpoint", "")
    g_host, g_port = host_port(green_ep)
    green_cmd = data.get("green_agent", {}).get("cmd", "")

    participants = []

    for p in data.get("participants", []):
        if isinstance(p, dict) and "endpoint" in p:
            h, pt = host_port(p["endpoint"])
            participants.append(
                {
                    "name": str(p.get("name", "")),
                    "host": h,
                    "port": pt,
                    "cmd": p.get("cmd", ""),
                }
            )

    cfg = data.get("config", {})

    return {
        "green_agent": {"host": g_host, "port": g_port, "cmd": green_cmd},
        "participants": participants,
        "config": cfg,
    }


def main():
    parser = argparse.ArgumentParser(description="Run agent scenario")
    parser.add_argument("scenario", help="Path to scenario TOML file")
    parser.add_argument(
        "--show-logs", action="store_true", help="Show agent stdout/stderr"
    )
    parser.add_argument(
        "--serve-only",
        action="store_true",
        help="Start agent servers only without running evaluation",
    )
    args = parser.parse_args()

    cfg = parse_toml(args.scenario)

    sink = None if args.show_logs or args.serve_only else subprocess.DEVNULL
    parent_bin = str(Path(sys.executable).parent)
    base_env = os.environ.copy()
    base_env["PATH"] = parent_bin + os.pathsep + base_env.get("PATH", "")

    procs = []
    try:
        # Start participant agent
        part_cmd_args = shlex.split(cfg["participants"][0].get("cmd", ""))
        if part_cmd_args:
            logger.info(
                (
                    f"Starting participant agent at {cfg['participants'][0]['host']}"
                    f":{cfg['participants'][0]['port']}"
                )
            )
            procs.append(
                subprocess.Popen(
                    part_cmd_args,
                    env=base_env,
                    stdout=sink,
                    stderr=sink,
                    text=True,
                )
            )

        # Start green agent
        green_cmd_args = shlex.split(cfg["green_agent"].get("cmd", ""))
        if green_cmd_args:
            logger.info(
                (
                    f"Starting green agent at {cfg['green_agent']['host']}"
                    f":{cfg['green_agent']['port']}"
                )
            )
            procs.append(
                subprocess.Popen(
                    green_cmd_args,
                    env=base_env,
                    stdout=sink,
                    stderr=sink,
                    text=True,
                )
            )

        # Wait for all agents to be ready
        if not asyncio.run(wait_for_agents(cfg)):
            logger.error("Error: Not all agents became ready. Exiting.")
            return

        logger.info("Agents started. Press Ctrl+C to stop.")

        if args.serve_only:
            while True:
                for proc in procs:
                    if proc.poll() is not None:
                        print(f"Agent exited with code {proc.returncode}")
                        break
                    time.sleep(0.5)
        else:
            client_proc = subprocess.Popen(
                [sys.executable, "-m", "ts_bench.client_cli", args.scenario],
                env=base_env,
            )
            procs.append(client_proc)
            client_proc.wait()

    except KeyboardInterrupt:
        pass

    finally:
        print("\nShutting down...")
        for p in procs:
            kill_process_tree(p)


if __name__ == "__main__":
    main()
