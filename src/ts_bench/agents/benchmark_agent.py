import argparse
import asyncio
import contextlib
import json
import logging
import os
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import requests
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    InvalidParamsError,
    TaskState,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
from openai import OpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

client = OpenAI()


@dataclass
class TaskParams:
    root_dir: str
    task_description: str
    eval_fn_path: str
    train_x: str
    train_y: str
    val_x: str
    val_y: str
    test_x: str
    target_shape: list[int]


def download_and_parse_timeseries_dataset(
    dir_path: str, dataset_url: str, output_shape: list[int]
) -> TaskParams:
    """
    Downloads a task dataset ZIP file (from a dataset API download URL),
    extracts it, parses task information, loads npy data files, and returns
    a structured dictionary.

    Expected folder structure inside ZIP:
        task.txt
        eval_function.py
        dataset/
            train_X.npy
            train_Y.npy
            val_X.npy
            val_Y.npy
            test_X.npy
    """
    logger.info(f"Downloading dataset from: {dataset_url}")

    # headers = {
    #     "User-Agent": (
    #         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    #         "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    #     )
    # }
    response = requests.get(dataset_url, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            (
                f"Failed to download dataset (status {response.status_code}): "
                "{response.text}"
            )
        )

    zipfile.ZipFile(BytesIO(response.content)).extractall(dir_path)

    task_txt_path = os.path.join(dir_path, "task.txt")

    if not os.path.exists(task_txt_path):
        raise FileNotFoundError("task.txt not found in extracted dataset.")

    with open(task_txt_path, "r") as f:
        task_description = f.read()

    eval_fn_path = os.path.join(dir_path, "eval_fn.py")

    if not os.path.exists(eval_fn_path):
        raise FileNotFoundError("eval_fn.py not found in dataset.")

    data_dir = os.path.join(dir_path, "dataset")

    return TaskParams(
        root_dir=dir_path,
        task_description=task_description,
        eval_fn_path=eval_fn_path,
        train_x=os.path.join(data_dir, "train_X.npy"),
        train_y=os.path.join(data_dir, "train_Y.npy"),
        val_x=os.path.join(data_dir, "val_X.npy"),
        val_y=os.path.join(data_dir, "val_Y.npy"),
        test_x=os.path.join(data_dir, "test_X.npy"),
        target_shape=output_shape,
    )


async def generate_solver_code(task_id: str, task_params: TaskParams) -> str:
    """
    Ask OpenAI to write Python code that:
    - trains a model
    - outputs a npy file of predictions
    """

    prompt = f"""
You are a Python time-series ML engineer.
You are given a task with the description: {task_params.task_description}
You are provided with dataset file paths:
- train_X: {task_params.train_x}
- train_Y: {task_params.train_y}
- val_X: {task_params.val_x}
- val_Y: {task_params.val_y}
- test_X: {task_params.test_x}

Write Python code that:

1. Loads training/validation sets.
2. Trains a model (any reasonable baseline).
3. Predicts the test set.
4. Saves a npy file containing predictions in this exact path:
   {task_params.root_dir}/{task_id}.npy

ONLY the following libraries are available for import:

- sklearn
- pandas
- numpy

Output ONLY runnable Python code, nothing else.
"""

    logger.info(f"Requesting OpenAI to generate solver code for task: {task_id}")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    code = response.choices[0].message.content
    # Strip Markdown fences if present
    if "```" in code:
        code = code.split("```")[1]
        code = code.replace("python", "").strip()

    return code


async def run_generated_code(tmp_dir: str, code: str, task_id: str) -> str:
    """Execute generated Python code in a temp file."""
    tmp_file = os.path.join(tmp_dir, f"{task_id}_solver.py")

    with open(tmp_file, "w") as f:
        f.write(code)

    logger.info(f"Executing solver code for task {task_id}...")

    proc = subprocess.run(["python", tmp_file], capture_output=True, text=True)

    if proc.returncode != 0:
        logger.error(f"Solver for {task_id} failed: \n{proc.stderr}")
        raise RuntimeError(proc.stderr)

    out_path = f"{tmp_dir}/{task_id}.npy"
    logger.info(f"Solved task {task_id}. Predictions saved to {out_path}")

    return out_path


class BaselineExecutorExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg_obj = context.get_user_input()

        logger.info(f"Participant agent received new task {context.task_id}")
        msg = context.message

        if msg is None:
            raise ServerError(error=InvalidParamsError(message="Missing message."))

        task = new_task(msg)
        await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                "Received assignment. Parsing tasks and generating solutions...",
                context_id=context.context_id,
            ),
        )

        msg_obj = json.loads(msg_obj)
        task_id = msg_obj["task_specification"]["task_id"]
        data_url = msg_obj["task_specification"]["url"]
        target_shape = msg_obj["task_specification"]["output_shape"]

        tmp_dir = tempfile.gettempdir()

        task_params = download_and_parse_timeseries_dataset(
            tmp_dir, data_url, target_shape
        )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Generating solver for task '{task_id}'...",
                context_id=context.context_id,
            ),
        )

        solver_code = await generate_solver_code(
            task_id=task_id, task_params=task_params
        )

        try:
            output_path = await run_generated_code(tmp_dir, solver_code, task_id)
            preds = np.load(output_path)
        except Exception as e:
            logger.error(f"Error running solver for task {task_id}: {e}")
            logger.warning("Using dummy predictions instead.")
            preds = np.random.randn(*target_shape)

        payload = {"predictions": preds.tolist()}

        final_message = json.dumps(payload)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                final_message,
                context_id=context.context_id,
            ),
        )

        await event_queue.enqueue_event(
            new_agent_text_message(final_message, context_id=task.context_id)
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


async def main():
    parser = argparse.ArgumentParser(description="Run benchmark participant agent.")
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the server"
    )
    parser.add_argument(
        "--port", type=int, default=9019, help="Port to bind the server"
    )
    parser.add_argument(
        "--card-url", type=str, help="External URL to provide in the agent card"
    )
    args = parser.parse_args()

    agent_url_cm = contextlib.nullcontext(
        args.card_url or f"http://{args.host}:{args.port}/"
    )

    async with agent_url_cm as agent_url:
        executor = BaselineExecutorExecutor()
        agent_card = AgentCard(
            name="TS-Bench Baseline Purple Agent",
            description="Baseline Time series ML problem solver",
            url=agent_url,
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=False),
            skills=[],
            supports_authenticated_extended_card=True,
        )

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )

        uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())
