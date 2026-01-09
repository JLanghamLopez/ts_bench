import argparse
import asyncio
import contextlib
import json
import logging
import os
import re
import subprocess
import tempfile
import zipfile
from typing import Dict, List

import numpy as np
import pandas as pd
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


# -----------------------------------------------------------
# Utility: Extract tasks from assignment message
# -----------------------------------------------------------
def parse_tasks_from_message(message: Dict) -> List[Dict]:
    """
    Extract task_id and data_url from the JSON message.
    """
    tasks = []
    current = {}

    current["task_id"] = message["task_specification"]["task_id"]
    current["data_url"] = message["task_specification"]["url"]
    if current:
        tasks.append(current)

    return tasks


def download_and_parse_kaggle_timeseries_dataset(dataset_url: str) -> dict:
    """
    Downloads a Kaggle dataset ZIP file (from a dataset API download URL),
    extracts it, parses task information, loads PKL data files, and returns
    a structured dictionary.

    Expected folder structure inside ZIP:
        task.txt
        eval_function.py
        dataset/
            train_X.pkl
            train_Y.pkl
            val_X.pkl
            val_Y.pkl
            test_X.pkl
    """

    tmp_dir = tempfile.mkdtemp(prefix="tsbench_")
    zip_path = os.path.join(tmp_dir, "dataset.zip")

    print(f"Downloading dataset from Kaggle with url: {dataset_url} to: {zip_path}")

    response = requests.get(dataset_url, stream=True)

    if response.status_code != 200:
        raise RuntimeError(f"Failed to download Kaggle dataset: {response.text}")

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp_dir)

    task_txt_path = os.path.join(tmp_dir, "task.txt")
    if not os.path.exists(task_txt_path):
        raise FileNotFoundError("task.txt not found in extracted dataset.")

    with open(task_txt_path, "r") as f:
        task_description = f.read()

    # get the target shape from the task description
    match = re.search(r"\[ *(\d+) *, *(\d+) *\]", task_description)
    if match:
        t, c = map(int, match.groups())

    eval_fn_path = os.path.join(tmp_dir, "eval_fn.py")
    if not os.path.exists(eval_fn_path):
        raise FileNotFoundError("eval_fn.py not found in dataset.")

    data_dir = os.path.join(tmp_dir, "dataset")

    def load_pkl(name):
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} missing in dataset/")
        return pd.read_pickle(path)

    bs = load_pkl("test_X.pkl").shape[0]

    return {
        "root_dir": tmp_dir,
        "task_description": task_description,
        "eval_fn_path": eval_fn_path,
        "train_X": os.path.join(data_dir, "train_X.pkl"),
        "train_Y": os.path.join(data_dir, "train_Y.pkl"),
        "val_X": os.path.join(data_dir, "val_X.pkl"),
        "val_Y": os.path.join(data_dir, "val_Y.pkl"),
        "test_X": os.path.join(data_dir, "test_X.pkl"),
        "target_shape": (bs, t, c),
    }


# -----------------------------------------------------------
# Utility: Call OpenAI to generate Python code
# -----------------------------------------------------------
async def generate_solver_code(
    task_id: str,
    task_description: str,
    train_X: str,
    train_Y: str,
    val_X: str,
    val_Y: str,
    test_X: str,
) -> str:
    """
    Ask OpenAI to write Python code that:
    - trains a model
    - outputs a npy file of predictions
    """

    prompt = f"""
You are a Python time-series ML engineer. You are given a task with the description: {task_description}
You are provided with dataset file paths:
- train_X: {train_X}
- train_Y: {train_Y}
- val_X: {val_X}
- val_Y: {val_Y}
- test_X: {test_X}

Write Python code that:

1. Loads training/validation sets.
2. Trains a model (any reasonable baseline).
3. Predicts the test set.
4. Saves a npy file containing predictions in this exact path:
   /tmp/{task_id}.npy

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


# -----------------------------------------------------------
# Utility: Execute generated code
# -----------------------------------------------------------
async def run_generated_code(code: str, task_id: str):
    """Execute generated Python code in a temp file."""
    tmp_file = f"/tmp/{task_id}_solver.py"
    with open(tmp_file, "w") as f:
        f.write(code)

    logger.info(f"Executing solver code for task {task_id}...")

    proc = subprocess.run(["python", tmp_file], capture_output=True, text=True)

    if proc.returncode != 0:
        logger.error(f"Solver for {task_id} failed:\n{proc.stderr}")
        raise RuntimeError(proc.stderr)

    logger.info(f"Solved task {task_id}. Predictions saved to /tmp/{task_id}.csv")


# -----------------------------------------------------------
# Purple Agent Executor
# -----------------------------------------------------------
class BaselineExecutorExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg_obj = context.get_user_input()

        logger.info(f"Participant agent received raw user_input object: {msg_obj}")
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

        # --------------------------------------------------
        # STEP 1 — Parse incoming JSON message from TSTaskAgent
        # --------------------------------------------------
        task_defs = parse_tasks_from_message(json.loads(msg_obj))

        # --------------------------------------------------
        # STEP 2 — For each task, generate solver code via OpenAI
        # --------------------------------------------------

        for t in task_defs:
            task_id = t["task_id"]
            data_url = t["data_url"]
            parsed_data = download_and_parse_kaggle_timeseries_dataset(data_url)
            task_description = parsed_data["task_description"]
            target_shape = parsed_data["target_shape"]
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"Generating solver for task '{task_id}'...",
                    context_id=context.context_id,
                ),
            )

            solver_code = await generate_solver_code(
                task_id=task_id,
                task_description=task_description,
                train_X=parsed_data["train_X"],
                train_Y=parsed_data["train_Y"],
                val_X=parsed_data["val_X"],
                val_Y=parsed_data["val_Y"],
                test_X=parsed_data["test_X"],
            )

            try:
                await run_generated_code(solver_code, task_id)
                # load predictions
                preds = np.load(f"/tmp/{task_id}.npy")
            except Exception as e:
                logger.error(f"Error running solver for task {task_id}: {e}")
                logger.warning("Using dummy predictions instead.")
                # create dummy predictions
                preds = np.random.randn(*target_shape)

            pay_load = {"predictions": preds.tolist()}

        # --------------------------------------------------
        # STEP 3 — Return JSON mapping
        # --------------------------------------------------

        final_message = json.dumps(pay_load)

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
