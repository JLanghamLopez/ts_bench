from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import os
from typing import Optional
from urllib.request import urlretrieve

import numpy as np
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import Part, TextPart
from a2a.utils import new_agent_text_message
from dotenv import load_dotenv

from ts_bench.agents.agent_card import ts_task_agent_card
from ts_bench.agents.base_agent import GreenAgent
from ts_bench.executor import TSBenchExecutor
from ts_bench.experiment_types import EvalRequest
from ts_bench.task_bank import Task, TaskBank, TaskDefinition, TaskType
from ts_bench.tool_provider import ToolProvider

from .eval_forecasting import eval_forecasting
from .eval_generation import eval_generation
from .evaluation import (
    TaskResult,
    _compute_score,
    aggregate_scores,
    failed_result,
)
from .task import AssignmentMessage, create_assignment_message
from .utils import validate_inputs

logger = logging.getLogger(__name__)

ALLOWED_TASK_TYPES: set[TaskType] = {
    TaskType.TIME_SERIES_FORECASTING,
    TaskType.TIME_SERIES_GENERATION,
}

METRICS_BY_TYPE: dict[TaskType, list[str]] = {
    TaskType.TIME_SERIES_FORECASTING: ["rmse", "mae", "mape"],
    TaskType.TIME_SERIES_GENERATION: ["histloss", "auto_corr", "cross_corr"],
}


class TSTaskAgent(GreenAgent):
    """
    Green agent for time-series benchmark tasks.

    Input:
        EvalRequest with config["task_type"] in
        {"time-series-forecasting", "time-series-generation"}

    Responsibilities:
    - Task assignment:
        Retrieves ALL tasks of that type from the TaskBank
        Constructs a structured assignment message and sends it to the participant.
        The participant agent must return a JSON object mapping:
        { task_id: "/path/to/predictions.csv", ... }
    - Task evaluation:
        After receiving prediction paths, the agent loads each task's predictions,
        And run the appropriate evaluation script and report scores/feedback.
    """

    def __init__(self, task_bank: TaskBank):
        self.task_bank = task_bank
        self._tool_provider = ToolProvider()
        self.ground_truth_key = os.environ["GROUND_TRUTH_KEY"]
        self.ground_truth_url = os.environ["GROUND_TRUTH_URL"]

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        """
        Workflow:
        1. Read task_type from request
        2. Fetch all tasks of that type from TaskBank
        3. In turn send each task as a textual instructions to purple agent
        4. Wait for purple agent to return predictions as JSON
        5. Run correct_fn.py to evaluate predictions
        6. Return evaluation results
        """

        logger.info("TSTaskAgent.run_eval started with request: %s", request)

        task_type_str: str = request.config.get("task_type", "")
        task_type = TaskType(task_type_str)

        # Fetch tasks and prepare assignment
        await updater.start_work(
            new_agent_text_message(
                f"Preparing task assignment for type='{task_type.value}'.",
                context_id=updater.context_id,
            ),
        )

        assignments: list[Task] = self.task_bank.get_tasks_by_type(task_type)

        results = []

        for i, task in enumerate(assignments):
            task_def = task.task_definition

            assignment_msg: AssignmentMessage = create_assignment_message(i, task_def)

            logger.info(
                f"Assigning task {i}: {task_def.name} for type={task_type.value}. "
                f"Sending instructions to participant agent."
            )

            await updater.start_work(
                new_agent_text_message(
                    (
                        f"Sending task {i}: {task_def.name} to "
                        f"participant agent for type='{task_type.value}'."
                    ),
                    context_id=updater.context_id,
                ),
            )

            logger.debug(
                "Assignment Message: \n%s", assignment_msg.model_dump_json(indent=2)
            )

            # Send to purple agent and wait for predictions
            try:
                participant_url = str(request.participant)
                new_conversation = i == 0
                response = await self._tool_provider.talk_to_agent(
                    message=assignment_msg.model_dump_json(indent=2),
                    url=participant_url,
                    new_conversation=new_conversation,
                )

                try:
                    parsed = json.loads(response)
                except Exception as e:
                    raise ValueError(f"Response is not valid JSON: {response}") from e

                result = np.array(parsed["predictions"], dtype=np.float32)
                logging.info(
                    f"Received results for task {task_def.name} with shape: {result.shape}"
                )

                await updater.start_work(
                    new_agent_text_message(
                        "Received predictions from participant. Starting evaluation.",
                        context_id=updater.context_id,
                    ),
                )

                try:
                    ground_truth_url = (
                        f"{self.ground_truth_url}{task.ground_truth_file}"
                        f"?{self.ground_truth_key}"
                    )
                    evaluation_result = await self._evaluate_predictions(
                        predictions=result,
                        assignment=task_def,
                        ground_truth_url=ground_truth_url,
                    )

                except Exception as e:
                    msg = f"Evaluation failed for task number {i}': {e}"
                    logger.error(msg, exc_info=True)
                    await updater.start_work(
                        new_agent_text_message(msg, context_id=updater.context_id),
                    )
                    evaluation_result = failed_result(task_def)

                results.append(evaluation_result)

                logger.info(
                    "Evaluation complete for task %d: %s\n"
                    "Score: %.2f/10\n"
                    "Evaluation Summary:\n%s",
                    i,
                    task_def.name,
                    evaluation_result.score,
                    evaluation_result.model_dump_json(indent=2),
                )

                await updater.start_work(
                    new_agent_text_message(
                        f"Evaluation complete for task {i}: {task_def.name} "
                        f"Score: {evaluation_result.score:.2f}/10\n\n"
                        f"Evaluation Summary: \n{evaluation_result.model_dump_json()}",
                        context_id=updater.context_id,
                    ),
                )

            except Exception as e:
                msg = f"Failed to communicate with participant agent or parse response: {e}"
                logger.error(msg, exc_info=True)
                raise RuntimeError(msg) from e

        summary = await aggregate_scores(task_type, results)

        logger.info(
            "About to add artifact and complete task.\n"
            "Task type: %s\n"
            "Number of results: %d\n"
            "Aggregated Summary:\n%s",
            task_type.value,
            len(results),
            summary.model_dump_json(indent=2),
        )

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=summary.model_dump_json()))],
            name="Result",
        )
        self._tool_provider.reset()

        return

    async def _evaluate_predictions(
        self,
        predictions: np.ndarray,
        assignment: TaskDefinition,
        ground_truth_url: str,
    ) -> Optional[TaskResult]:
        """
        Run the correct_fn evaluation for the given task_type.

        - Call task-specific eval_fn to compute raw_metrics
        - Compute primary metric normalized score and difficulty-weighted score
        """
        task_type = assignment.task_type
        logger.info("Running evaluation for task_type='%s'.", task_type.value)

        logger.info(
            "Evaluating task_id='%s', difficulty='%s'",
            assignment.task_id,
            assignment.difficulty.value,
        )

        path, _ = urlretrieve(ground_truth_url)
        gt_tensor = np.load(path, allow_pickle=False)

        valid, err = validate_inputs(task_type, predictions, gt_tensor)
        if not valid:
            raise ValueError(
                f"Validation failed for task_id={assignment.task_id}: {err}"
            )

        eval_fn = (
            eval_forecasting
            if task_type == TaskType.TIME_SERIES_FORECASTING
            else eval_generation
        )

        # run the evaluation
        raw_metrics = eval_fn(predictions, gt_tensor)

        # compute average normalized score
        score = _compute_score(raw_metrics)

        return TaskResult(
            task_id=assignment.task_id,
            name=assignment.name,
            description=assignment.description,
            difficulty=assignment.difficulty,
            raw_metrics=raw_metrics,
            score=score,
        )

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """
        Validate EvalRequest: check that task_type is provided and tasks exist.
        """
        config = request.config or {}
        task_type_str = config.get("task_type")

        if not isinstance(task_type_str, str) or not task_type_str.strip():
            return False, "'task_type' must be a non-empty string."

        try:
            task_type = TaskType(task_type_str)
        except ValueError:
            return (
                False,
                f"'task_type' must be one of {[t.value for t in TaskType]}",
            )

        tasks = self.task_bank.get_tasks_by_type(task_type)
        if not tasks:
            return False, f"No tasks available for task_type='{task_type.value}'."

        return True, "Valid request."


async def main():
    parser = argparse.ArgumentParser(
        description="Run the time-series task generation agent."
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the server"
    )
    parser.add_argument(
        "--port", type=int, default=9019, help="Port to bind the server"
    )
    parser.add_argument(
        "--card-url", type=str, help="External URL to provide in the agent card"
    )
    parser.add_argument(
        "--tasks-path",
        type=str,
        default="./data/tasks.yaml",
        help="Path to task description YAML file",
    )
    args = parser.parse_args()

    agent_url_cm = contextlib.nullcontext(
        args.card_url or f"http://{args.host}:{args.port}/"
    )
    tasks_yaml_path = args.tasks_path

    async with agent_url_cm as agent_url:
        logger.info(f"Loading tasks from {tasks_yaml_path}")
        task_bank = TaskBank(tasks_yaml_path)
        logger.info("TaskBank initialised with %d tasks.", task_bank.loaded_tasks)
        green_agent = TSTaskAgent(task_bank)

        executor = TSBenchExecutor(green_agent)
        agent_card = ts_task_agent_card(url=agent_url)

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
    load_dotenv()
    asyncio.run(main())
