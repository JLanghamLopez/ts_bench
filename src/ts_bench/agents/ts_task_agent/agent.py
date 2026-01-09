from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import Part, TextPart
from a2a.utils import new_agent_text_message

from data.task_bank import TaskBank, TaskDefinition, TaskType
from ts_bench.agents.agent_card import ts_task_agent_card
from ts_bench.agents.base_agent import GreenAgent
from ts_bench.executor import TSBenchExecutor
from ts_bench.experiment_types import EvalRequest
from ts_bench.tool_provider import ToolProvider

from .eval_fn_combined import eval_forecasting, eval_generation
from .evaluation import (
    TaskResult,
    _compute_score,
    aggregate_scores,
    failed_result,
)
from .task import AssignmentMessage, create_assignment_message
from .utils import check_response, load_ground_truth, validate_inputs

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

    def __init__(
        self,
        task_bank: TaskBank,
        dataset_root: Optional[str | Path] = None,
        test_batch_size: Optional[int] = None,
    ):
        self.task_bank = task_bank
        self._tool_provider = ToolProvider()

        if dataset_root is None:
            file_dir = Path(__file__).resolve().parent
            proj_dir = file_dir.parents[3]
            dataset_root = proj_dir / "data/tasks"

        self.dataset_root = Path(dataset_root)

        self.test_batch_size = test_batch_size

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        """
        Workflow:
        1. Read task_type from request
        2. Fetch all tasks of that type from TaskBank
        3. In turn send each task as a textual instructions to purple agent
        4. Wait for purple agent to return prediction (path to .csv)
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

        assignments: list[TaskDefinition] = self.task_bank.get_tasks_by_type(task_type)

        results = []

        # Submit each task in turn
        for i, task in enumerate(assignments):
            # Create instruction message
            assignment_msg: AssignmentMessage = create_assignment_message(i, task)

            logger.info(
                f"Assigning task{i}: {task.name} for type={task_type.value}. "
                f"Sending instructions to participant agent."
            )

            await updater.start_work(
                new_agent_text_message(
                    (
                        f"Sending task {i}: {task.name} to "
                        f"participant agent for type='{task_type.value}'."
                    ),
                    context_id=updater.context_id,
                ),
            )

            logger.info(
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
                result = np.array(parsed["predictions"])
                logging.info(
                    f"Received results for task {task.name} with shape: {result.shape}"
                )

                await updater.start_work(
                    new_agent_text_message(
                        "Received prediction path from participant. Starting evaluation.",
                        context_id=updater.context_id,
                    ),
                )
                """
                TODO:
                - Rewrite the commented sections below and refactor the `evaluate_predictions` function
                to operate directly on NumPy array {result} rather than file paths.
                """
            #     try:
            #         evaluation_result = await self._evaluate_predictions(
            #             predictions_path=response,
            #             assignment=task,
            #         )

            #     except Exception as e:
            #         msg = f"Evaluation failed for task number {i}': {e}"
            #         logger.error(msg, exc_info=True)
            #         await updater.start_work(
            #             new_agent_text_message(msg, context_id=updater.context_id),
            #         )
            #         evaluation_result = failed_result(response, task)

            #     results.append(evaluation_result)

            #     logger.info(
            #         "Evaluation complete for task %d: %s\n"
            #         "Score: %.2f/10\n"
            #         "Evaluation Summary:\n%s",
            #         i,
            #         task.name,
            #         evaluation_result.score,
            #         evaluation_result.model_dump_json(indent=2),
            #     )

            #     # Return final evaluation results
            #     await updater.start_work(
            #         new_agent_text_message(
            #             f"Evaluation complete for task {i}: {task.name} "
            #             f"Score: {evaluation_result.score:.2f}/10\n\n"
            #             f"Evaluation Summary:\n{evaluation_result.model_dump_json()}",
            #             context_id=updater.context_id,
            #         ),
            #     )

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
        predictions_path: str,
        assignment: TaskDefinition,
    ) -> Optional[TaskResult]:
        """
        Run the correct_fn evaluation for the given task_type.
        This is a placeholder that will call the appropriate evaluation function.

        - Load predictions pkl from predictions[task_id]
        - Call task-specific eval_fn to compute raw_metrics
        - Compute primary metric normalized score and difficulty-weighted score
        """
        task_type = assignment.task_type
        logger.info("Running evaluation for task_type='%s'.", task_type.value)

        logger.info(
            "Evaluating task_id='%s', difficulty='%s', prediction_path='%s'",
            assignment.task_id,
            assignment.difficulty.value,
            predictions_path or "<missing>",
        )

        # load the prediction tensor
        pred_tensor = np.load(Path(predictions_path))

        # load ground truth tensor
        task_dir = self.dataset_root / assignment.task_id

        if task_type == TaskType.TIME_SERIES_FORECASTING:
            candidates = ["test_Y.npz", "test_Y.pkl"]
        else:
            candidates = ["test.npz", "test.pkl"]

        for name in candidates:
            candidate = task_dir / name
            if candidate.exists():
                gt_path = candidate
                break

        if not gt_path.exists():
            raise FileNotFoundError(
                f"Missing ground-truth file. Tried: {candidates} in {task_dir}"
            )

        gt_tensor = load_ground_truth(gt_path)

        # Validate the inputs (predictions and ground truth)
        valid, err = validate_inputs(task_type, pred_tensor, gt_tensor)
        if not valid:
            raise ValueError(
                f"Validation failed for task_id={assignment.task_id}: {err}"
            )

        # Choose the correct evaluation function based on task type
        eval_fn = (
            eval_forecasting
            if task_type == TaskType.TIME_SERIES_FORECASTING
            else eval_generation
        )

        # run the evaluation
        raw_metrics = eval_fn(pred_tensor, gt_tensor)

        # compute average normalized score
        score = _compute_score(raw_metrics)

        return TaskResult(
            task_id=assignment.task_id,
            name=assignment.name,
            description=assignment.description,
            difficulty=assignment.difficulty,
            raw_metrics=raw_metrics,
            score=score,
            prediction_path=predictions_path,
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
    args = parser.parse_args()

    agent_url_cm = contextlib.nullcontext(
        args.card_url or f"http://{args.host}:{args.port}/"
    )

    file_dir = Path(__file__).resolve().parent
    proj_dir = file_dir.parents[3]

    tasks_json_path = (proj_dir / "data/tasks/tasks.json").resolve()

    async with agent_url_cm as agent_url:
        task_bank = TaskBank(
            tasks_json_path=str(tasks_json_path),
        )
        logger.info("TaskBank initialised with %d tasks.", len(task_bank._tasks_by_id))

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
    asyncio.run(main())
