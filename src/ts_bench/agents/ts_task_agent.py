from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
from pathlib import Path

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message
from litellm import acompletion

from data.task_bank import TaskBank, TaskDefinition, TaskDifficulty, TaskType
from ts_bench.agents.agent_card import ts_task_agent_card
from ts_bench.agents.base_agent import GreenAgent
from ts_bench.executor import TSBenchExecutor
from ts_bench.experiment_types import EvalRequest
from ts_bench.tool_provider import ToolProvider

logger = logging.getLogger(__name__)

USE_LLM_FEEDBACK = False

ALLOWED_TASK_TYPES: set[TaskType] = {
    TaskType.TIME_SERIES_FORECASTING,
    TaskType.TIME_SERIES_GENERATION,
}

METRICS_BY_TYPE: dict[TaskType, list[str]] = {
    TaskType.TIME_SERIES_FORECASTING: ["rmse", "mae", "mape"],
    TaskType.TIME_SERIES_GENERATION: ["histloss", "auto_corr", "cross_corr"],
}

PRIMARY_METRIC: dict[TaskType, str] = {
    TaskType.TIME_SERIES_FORECASTING: "rmse",
    TaskType.TIME_SERIES_GENERATION: "histloss",
}

DIFFICULTY_WEIGHTS: dict[TaskDifficulty, float] = {
    TaskDifficulty.EASY: 1.0,
    TaskDifficulty.INTERMEDIATE: 1.0,
    TaskDifficulty.ADVANCED: 1.0,
}

# score = 1 / (1 + a * loss^b)
METRIC_NORMALIZATION = {
    "rmse": {"a": 1.0, "b": 1.0},
    "mae": {"a": 1.0, "b": 1.0},
    "mape": {"a": 1.0, "b": 1.0},
    "histloss": {"a": 1.0, "b": 1.0},
    "auto_corr": {"a": 1.0, "b": 1.0},
    "cross_corr": {"a": 1.0, "b": 1.0},
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
        Constructs a structured assignment message and sends it to the participant
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
        3. Send detailed textual instructions to purple agent
        4. Wait for purple agent to return predictions (path to .csv)
        5. Run correct_fn.py to evaluate predictions
        6. Return evaluation results
        """

        logger.info("TSTaskAgent.run_eval started with request: %s", request)

        task_type_str: str = request.config.get("task_type", "")
        task_type = TaskType(task_type_str)

        # Fetch tasks and prepare assignment
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Preparing task assignment for type='{task_type.value}'.",
                context_id=updater.context_id,
            ),
        )

        assignments: list[TaskDefinition] = self.task_bank.get_tasks_by_type(task_type)

        # Create instruction message
        assignment_msg = self._create_assignment_message(task_type, assignments)

        logger.info(
            f"Assigning {len(assignments)} tasks for type={task_type.value}. "
            f"Sending instructions to participant agent."
        )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                (
                    f"Sending {len(assignments)} task(s) to "
                    f"participant agent for type='{task_type.value}'."
                ),
                context_id=updater.context_id,
            ),
        )

        logger.info("Assignment Message: \n%s", assignment_msg)
        # Send to purple agent and wait for predictions
        try:
            participant_url = str(request.participant)
            raw_response = await self._tool_provider.talk_to_agent(
                message=assignment_msg,
                url=participant_url,
                new_conversation=True,
            )

            logger.info("Received response from participant agent: %s", raw_response)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    "Received prediction paths from participant. Starting evaluation.",
                    context_id=updater.context_id,
                ),
            )

            predictions_map = self._parse_predictions_mapping(raw_response, assignments)

        except Exception as e:
            msg = f"Failed to communicate with participant agent or parse response: {e}"
            logger.error(msg)
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(msg, context_id=updater.context_id),
            )
            raise

        # Run evaluation using desrired eval_fn
        try:
            evaluation_summary = await self._evaluate_predictions(
                task_type=task_type,
                predictions=predictions_map,
                assignments=assignments,
            )
        except Exception as e:
            msg = f"Evaluation failed for task_type='{task_type.value}': {e}"
            logger.error(msg, exc_info=True)
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(msg, context_id=updater.context_id),
            )
            raise

        # Return final evaluation results
        final_score = evaluation_summary["final_score_0_to_10"]

        logger.info(
            "Evaluation complete for task_type='%s'. Final score: %.2f/10",
            task_type.value,
            final_score,
        )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Evaluation complete for task_type='{task_type.value}'. "
                f"Final Score: {final_score:.2f}/10\n\n"
                f"Evaluation Summary:\n{json.dumps(evaluation_summary, indent=2)}",
                context_id=updater.context_id,
            ),
        )

    def _create_assignment_message(
        self, task_type: TaskType, assignments: list[TaskDefinition]
    ) -> str:
        """
        Create a textual instruction message for the purple agent.
        This tells them about the tasks and what's expected.
        """
        msg_lines = [
            f"# Time Series Benchmark - {task_type.replace('-', ' ').title()} Tasks",
            "",
            f"You have been assigned {len(assignments)} task(s) for evaluation.",
            "",
            "## Instructions",
            "",
            "For each task below, you will receive a URL to download the task bundle.",
            "Each bundle contains:",
            "- Training data",
            "- Validation data",
            "- Test data",
            "- Evaluation metrics code",
            "- Task description and requirements",
            "",
            "You are expected to:",
            "1. Download and analyze each task bundle",
            "2. Build and train and tune your model on the training and validation data",
            "3. Generate predictions for the test data for EACH task.",
            "4. Return a JSON object mapping task_id to the file path of your predictions CSV."
            "",
            "## Task List",
            "",
        ]

        for i, task in enumerate(assignments, 1):
            msg_lines.extend(
                [
                    f"### Task {i}: {task.name}",
                    f"- **Task ID**: {task.task_id}",
                    f"- **Type**: {task.task_type.value}",
                    f"- **Difficulty**: {task.difficulty.value}",
                    f"- **Data URL**: {task.url}",
                    "",
                ]
            )

        msg_lines.extend(
            [
                "## Submission Format",
                "",
                "Please return ONLY a single JSON object in the response",
                "with the following structure:",
                "",
                "```json",
                "{",
                '  "<task_id_1>": "/path/to/predictions_task_1.csv",',
                '  "<task_id_2>": "/path/to/predictions_task_2.csv"',
                "}",
                "```",
                "",
                "- Keys MUST be task_ids from the list above.",
                "- Values MUST be the corresponding CSV file paths on remote filesystem.",
                "- Each CSV should contain predictions ONLY for the corresponding task, in the proper format.",
                "",
            ]
        )

        return "\n".join(msg_lines)

    def _parse_predictions_mapping(
        self,
        raw_response: str,
        assignments: list[TaskDefinition],
    ) -> dict[str, str]:
        """
        Parse participant response as JSON mapping: {task_id: csv_path}.
        Only keep task_ids in the current assignment set.
        """

        try:
            clean = raw_response.strip()
            if clean.startswith("```"):
                clean = clean.strip("`")
            data = json.loads(clean)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Participant response is not valid JSON: {e}. "
                f"Expected a JSON object mapping task_id to CSV path."
            )

        if not isinstance(data, dict):
            raise ValueError(
                "Participant response must be a JSON object mapping task_id to CSV path."
            )

        valid_task_ids = {t.task_id for t in assignments}
        predictions: dict[str, str] = {}

        for task_id, path in data.items():
            if task_id not in valid_task_ids:
                logger.warning(
                    "Participant provided predictions for unknown task_id '%s'. "
                    "which will be ignored.",
                    task_id,
                )
                continue

            if not isinstance(path, str) or not path.strip():
                raise ValueError(
                    f"Invalid CSV path for task_id='{task_id}': {path!r}. "
                    "Must be a non-empty string."
                )

            predictions[task_id] = path.strip()

        missing = valid_task_ids - set(predictions.keys())
        if missing:
            logger.warning(
                "No predictions provided for the following assigned tasks: %s",
                sorted(missing),
            )

        if not predictions:
            raise ValueError("No valid task_id → CSV path mappings found in response.")

        return predictions

    async def _evaluate_predictions(
        self,
        task_type: TaskType,
        predictions: dict[str, str],
        assignments: list[TaskDefinition],
    ) -> dict:
        """
        Run the correct_fn evaluation for the given task_type.
        This is a placeholder that will call the appropriate evaluation function.

        - Load predictions CSV from predictions[task_id]
        - Call task-specific eval_fn (TODO) to compute raw_metrics
        - Compute primary metric normalized score and difficulty-weighted score
        """

        logger.info(
            "Running evaluation for task_type='%s' with %d tasks.",
            task_type.value,
            len(assignments),
        )

        metric_names = METRICS_BY_TYPE[task_type]
        primary_metric = PRIMARY_METRIC[task_type]

        per_task_evals: dict[str, dict] = {}

        for task in assignments:
            pred_path = predictions.get(task.task_id)
            logger.info(
                "Evaluating task_id='%s', difficulty='%s', prediction_path='%s'",
                task.task_id,
                task.difficulty.value,
                pred_path or "<missing>",
            )

            if not pred_path:
                # raw_metrics = {m: float("inf") for m in metric_names}
                if task_type is TaskType.TIME_SERIES_FORECASTING:
                    raw_metrics = {"rmse": 0.5, "mae": 0.4, "mape": 0.3}
                else:
                    raw_metrics = {"histloss": 0.6, "auto_corr": 0.5, "cross_corr": 0.4}
            else:
                # TODO: call correct eval_fn.py
                # raw_metrics = await self._run_single_task_eval(task, pred_path, metric_names)
                # Placeholder
                if task_type is TaskType.TIME_SERIES_FORECASTING:
                    raw_metrics = {"rmse": 0.5, "mae": 0.4, "mape": 0.3}
                else:
                    raw_metrics = {"histloss": 0.6, "auto_corr": 0.5, "cross_corr": 0.4}

            primary_eval = self._compute_primary_metric_score(
                task_type=task_type,
                difficulty=task.difficulty,
                metrics=raw_metrics,
            )

            per_task_evals[task.task_id] = {
                "task_id": task.task_id,
                "name": task.name,
                "description": task.description,
                "difficulty": task.difficulty.value,
                "raw_metrics": raw_metrics,
                "primary_eval": primary_eval,
                "prediction_path": pred_path,
            }

        # Aggregate scores
        weighted_scores = [
            v["primary_eval"]["weighted_score"] for v in per_task_evals.values()
        ]
        weights_sum = sum(
            v["primary_eval"]["difficulty_weight"] for v in per_task_evals.values()
        )

        overall_weighted_score = (
            sum(weighted_scores) / weights_sum if weights_sum > 0 else 0.0
        )
        final_score = max(0.0, min(10.0, 10.0 * overall_weighted_score))

        evaluation_summary = {
            "task_type": task_type.value,
            "primary_metric": primary_metric,
            "num_tasks": len(assignments),
            "per_task": per_task_evals,
            "overall_weighted_score_0_to_1": overall_weighted_score,
            "final_score_0_to_10": final_score,
            # for LLM feedback prompt
            "metric_normalization_params": METRIC_NORMALIZATION,
            "difficulty_weights": {d.value: w for d, w in DIFFICULTY_WEIGHTS.items()},
        }

        if USE_LLM_FEEDBACK:
            try:
                feedback = await self._generate_feedback(task_type, evaluation_summary)
                if feedback:
                    evaluation_summary["feedback"] = feedback
            except Exception as e:
                logger.warning("LLM feedback generation failed: %s", e)

        return evaluation_summary

    def _normalize_metric(
        self,
        metric_name: str,
        raw_value: float,
    ) -> float:
        """
        Map a raw non-negative loss-like metric value (lower is better)
        to a score in (0,1]: s = 1 / (1 + a * value^b)
        """

        # raw metric can not be negative
        if raw_value < 0:
            raise ValueError(
                f"Metric '{metric_name}' must be non-negative, " f"got {raw_value}."
            )

        cfg = METRIC_NORMALIZATION.get(metric_name)

        # default to a=b=1
        a = float(cfg.get("a", 1.0)) if cfg else 1.0
        b = float(cfg.get("b", 1.0)) if cfg else 1.0

        # s = 1 / (1 + a * value^b) in (0,1]
        s = 1.0 / (1.0 + a * (raw_value**b))
        return max(0.0, min(1.0, s))

    def _compute_primary_metric_score(
        self,
        task_type: TaskType,
        difficulty: TaskDifficulty,
        metrics: dict[str, float],
    ) -> dict:
        """
        Compute normalized + weighted score from the PRIMARY_METRIC.
        """
        primary = PRIMARY_METRIC[task_type]
        raw_value = float(metrics[primary])

        normalized_score = self._normalize_metric(primary, raw_value)
        weight = DIFFICULTY_WEIGHTS.get(difficulty, 1.0)
        weighted = normalized_score * weight

        return {
            "primary_metric": primary,
            "raw_loss": raw_value,
            "normalized_score": normalized_score,
            "difficulty_weight": weight,
            "weighted_score": weighted,
        }

    async def _generate_feedback(
        self, task_type: TaskType, summary: dict
    ) -> str | None:
        if not USE_LLM_FEEDBACK:
            return None

        primary = summary["primary_metric"]
        score = summary["final_score_0_to_10"]

        prompt = f"""
You are an expert evaluator for time-series machine learning models.

The task_type is: '{task_type.value}'.

PRIMARY METRIC FOR SCORING:
- {primary}

NORMALIZATION METHOD:
score = 1 / (1 + a * loss^b)
Normalization parameters per metric:
{json.dumps(summary["metric_normalization_params"], indent=2)}

DIFFICULTY WEIGHTS:
{json.dumps(summary["difficulty_weights"], indent=2)}

FINAL SCORE:
- {score:.2f} out of 10

PER-TASK DETAILS (including descriptions and ALL raw metrics):
{json.dumps(summary["per_task"], indent=2)}

Please provide:
1. A summary of the participant's strengths.
2. Weaknesses, focusing on trends in metrics.
3. Insights about performance across difficulties.
4. Actionable improvement suggestions.
5. DO NOT repeat raw numbers exactly — interpret them qualitatively.
6. Reference secondary metrics (MAE, MAPE, AutoCorr, CrossCorr)
   to support your reasoning.

ENSURE that your feedback is complete.
"""

        response = await acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.2,
        )

        msg = response["choices"][0]["message"]["content"]
        return msg if isinstance(msg, str) else json.dumps(msg)

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
