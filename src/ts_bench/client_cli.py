import asyncio
import json
import logging
import sys
import tomllib
from pathlib import Path

from a2a.types import (
    AgentCard,
    DataPart,
    Message,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
    TextPart,
)

from ts_bench.client import send_message
from ts_bench.experiment_types import EvalRequest

logger = logging.getLogger(__name__)


def parse_toml(d: dict[str, object]) -> tuple[EvalRequest, str]:
    green = d.get("green_agent")
    if not isinstance(green, dict) or "endpoint" not in green:
        raise ValueError("green.endpoint is required in TOML")

    green_endpoint: str = green["endpoint"]

    purple = d.get("participant")
    if not isinstance(purple, dict) or "endpoint" not in purple:
        raise ValueError("participant.endpoint is required in TOML")

    eval_req = EvalRequest(
        participant=purple["endpoint"], config=d.get("config", {}) or {}
    )
    return eval_req, green_endpoint


def print_parts(parts, task_state: str | None = None):
    text_parts = []
    data_parts = []

    for part in parts:
        if isinstance(part.root, TextPart):
            try:
                data_item = json.loads(part.root.text)
                data_parts.append(data_item)
            except Exception:
                text_parts.append(part.root.text.strip())
        elif isinstance(part.root, DataPart):
            data_parts.append(part.root.data)

    output = []

    if task_state:
        output.append(f"[Status: {task_state}]")
    if text_parts:
        output.append("\n".join(text_parts))
    if data_parts:
        output.extend(json.dumps(item, indent=2) for item in data_parts)

    logger.info("\n".join(output) + "\n")


async def event_consumer(event, card: AgentCard) -> None:
    match event:
        case Message() as msg:
            print_parts(msg.parts)

        case (task, TaskStatusUpdateEvent() as status_event):
            status = status_event.status
            parts = status.message.parts if status.message else []
            print_parts(parts, status.state.value)
            if status.state.value == "completed":
                logger.info(task.artifacts)

        case (task, TaskArtifactUpdateEvent() as artifact_event):
            print_parts(artifact_event.artifact.parts, "Artifact update")

        case task, None:
            status = task.status
            parts = status.message.parts if status.message else []
            print_parts(parts, task.status.state.value)

        case _:
            logger.info("Unhandled event")


async def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python client_cli.py <scenario.toml>")
        sys.exit(1)

    path = Path(sys.argv[1])

    if not path.exists():
        logger.error(f"File not found: {path}")
        sys.exit(1)

    toml_data = path.read_text()
    data = tomllib.loads(toml_data)

    req, green_url = parse_toml(data)

    msg = req.model_dump_json()
    await send_message(msg, green_url, streaming=False, consumer=event_consumer)


if __name__ == "__main__":
    asyncio.run(main())
