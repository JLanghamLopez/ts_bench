from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
from pathlib import Path

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

from .evaluation import aggregate_scores, evaluate_predictions, failed_result
from .task import create_assignment_message
from .utils import check_response

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
            assignment_msg = create_assignment_message(i, task)

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

            logger.info("Assignment Message: \n%s", assignment_msg)

            # Send to purple agent and wait for predictions
            try:
                participant_url = str(request.participant)
                new_conversation = i == 0
                response = await self._tool_provider.talk_to_agent(
                    message=assignment_msg,
                    url=participant_url,
                    new_conversation=new_conversation,
                )

                logger.info("Received response from participant agent: %s", response)

                check_response(response)
                await updater.start_work(
                    new_agent_text_message(
                        "Received prediction path from participant. Starting evaluation.",
                        context_id=updater.context_id,
                    ),
                )
                try:
                    evaluation_result = await evaluate_predictions(
                        predictions_path=response,
                        assignment=task,
                    )
                    logger.info(
                        "Evaluation complete for task_type='%s'. Final score: %.2f/10",
                        task_type.value,
                        evaluation_result.primary_eval,
                    )

                except Exception as e:
                    msg = f"Evaluation failed for task number {i}': {e}"
                    logger.error(msg, exc_info=True)
                    await updater.start_work(
                        new_agent_text_message(msg, context_id=updater.context_id),
                    )
                    evaluation_result = failed_result(response, task)

                results.append(evaluation_result)

                # Return final evaluation results
                await updater.start_work(
                    new_agent_text_message(
                        f"Evaluation complete for task {i}: {task.name} "
                        f"Score: {evaluation_result.primary_eval:.2f}/10\n\n"
                        f"Evaluation Summary:\n{evaluation_result.model_dump_json()}",
                        context_id=updater.context_id,
                    ),
                )

            except Exception as e:
                msg = f"Failed to communicate with participant agent or parse response: {e}"
                logger.error(msg)
                # TODO: Should we write a better exception here?
                raise e

        summary = await aggregate_scores(task_type, results)

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=summary.model_dump_json()))],
            name="Result",
        )
        self._tool_provider.reset()

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
    proj_dir = file_dir.parents[2]

    tasks_json_path = (proj_dir / "data/tasks.json").resolve()

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
