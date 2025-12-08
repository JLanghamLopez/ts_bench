import argparse
import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import TaskState
from a2a.utils import new_agent_text_message
from litellm import acompletion

from ts_bench.agents.agent_card import ts_task_agent_card
from ts_bench.agents.base_agent import GreenAgent
from ts_bench.agents.task_bank import TaskBank, TaskDefinition
from ts_bench.executor import TSBenchExecutor
from ts_bench.experiment_types import EvalRequest, TaskAssignment

logger = logging.getLogger(__name__)

USE_LLM_FEEDBACK = True

ALLOWED_TASK_TYPES = {
    "time-series-forecasting",
    "time-series-generation",
}

METRICS_BY_TYPE: dict[str, list[str]] = {
    "time-series-forecasting": ["rmse", "mae", "quantile_loss"],
    "time-series-generation": ["sigw1", "auto_corr", "cross_corr"],
}

PRIMARY_METRIC: dict[str, str] = {
    "time-series-forecasting": "rmse",
    "time-series-generation": "sigw1",
}

DIFFICULTY_WEIGHTS: dict[str, float] = {
    "Easy": 1.0,
    "Intermediate": 1.0,
    "Advanced": 1.0,
}

# score = 1 / (1 + a * loss^b)
METRIC_NORMALIZATION = {
    "rmse": {"a": 1.0, "b": 1.0},
    "mae": {"a": 1.0, "b": 1.0},
    "quantile_loss": {"a": 1.0, "b": 1.0},
    "sigw1": {"a": 1.0, "b": 1.0},
    "auto_corr": {"a": 1.0, "b": 1.0},
    "cross_corr": {"a": 1.0, "b": 1.0},
}


class TSTaskAgent(GreenAgent):
    """
    Green agent for time-series benchmark tasks.

    Responsibilities:
    - Task assignment:
        Input: EvalRequest with config["task_type"] in
               {"forecasting", "generative-modelling"}
        Return ALL tasks of that type (various difficulties) to the purple agent.
    - Task evaluation:
        Input: EvalRequest with config["task_id"] and predictions
               (e.g. predictions.csv).
        Run the appropriate evaluation script and report scores/feedback.
    """

    def __init__(self, task_bank: TaskBank):
        self.task_bank = task_bank

    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        logger.info("TSTaskAgent.run_eval started with request: %s", request)

        # Check if it is task assignment request
        if self._is_assignment_request(request):
            await self._handle_task_assignment(request, updater)
            return

        # Check if it is task evaluation request
        if self._is_evaluation_request(request):
            await self._handle_task_evaluation(request, updater)
            return

        # Fail
        msg = (
            "Invalid request: must contain either "
            "'task_type' for task assignment or task_type + results for evaluation."
        )

        logger.warning("TSTaskAgent.run_eval: %s", msg)

        await updater.update_status(
            TaskState.failed,
            new_agent_text_message(msg, context_id=updater.context_id),
        )
        raise ValueError(msg)

    async def _handle_task_assignment(
        self,
        request: EvalRequest,
        updater: TaskUpdater,
    ) -> None:
        """
        - Reads task_type from request.
        - Fetches one/all tasks of that type from TaskBank.
        - Returns them as TaskAssignment objects.
        """

        task_type: str = request.config["task_type"]

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Assigning time-series tasks for type='{task_type}'.",
                context_id=updater.context_id,
            ),
        )

        tasks: list[TaskDefinition] = self.task_bank.get_tasks_by_type(task_type)

        if not tasks:
            msg = f"No tasks available for task_type='{task_type}'."
            logger.warning(msg)
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(msg, context_id=updater.context_id),
            )
            raise ValueError(msg)

        assignments: list[TaskAssignment] = []

        for t in tasks:
            data_url = self.task_bank.get_presigned_url(
                t.data_s3_key,
                self.task_bank.s3_bucket,
            )
            eval_fn_url = self.task_bank.get_presigned_url(
                t.eval_fn_s3_key,
                self.task_bank.s3_bucket,
            )

            assignment = TaskAssignment(
                task_id=t.task_id,
                name=t.name,
                description=t.description,
                task_type=t.task_type,
                difficulty=t.difficulty,
                data_url=data_url,
                eval_fn_url=eval_fn_url,
            )
            assignments.append(assignment)

        payload = [a.model_dump(mode="json") for a in assignments]

        logger.info(
            "Assigned %d tasks for task_type='%s' to participant=%s. Details: %s",
            len(assignments),
            task_type,
            request.participant,
            json.dumps(payload, indent=2),
        )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Assigned {len(assignments)} tasks for type='{task_type}'. "
                f"Details: {json.dumps(payload)}",
                context_id=updater.context_id,
            ),
        )

    async def _handle_task_evaluation(
        self,
        request: EvalRequest,
        updater: TaskUpdater,
    ) -> None:
        """
        For each task_id of that type:
            take raw loss-like metric values
            normalize per metric to [0,1]
            average to per-task score
            apply difficulty weight
        Aggregate per-task weighted scores into a final score in [0,10].
        Call LLM to generate feedback.
        """

        config = request.config
        task_type = config["task_type"]
        results = config["results"]

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting evaluation for task_type='{task_type}'.",
                context_id=updater.context_id,
            ),
        )

        # get the list of tasks for this type
        registered_tasks = self.task_bank.get_tasks_by_type(task_type)
        tasks_by_id = {t.task_id: t for t in registered_tasks}

        # submissions: {task_id: metrics}
        submitted_metrics: dict[str, dict[str, float]] = {
            item["task_id"]: {k: float(v) for k, v in item["metrics"].items()}
            for item in results
        }

        per_task_evals: dict[str, dict[str, object]] = {}
        weighted_scores: list[float] = []
        weights_sum: float = 0.0

        for tid, task_def in tasks_by_id.items():
            metrics_for_task = submitted_metrics[tid]

            primary_eval = self._compute_primary_metric_score(
                task_type=task_type,
                difficulty=task_def.difficulty,
                metrics=metrics_for_task,
            )

            per_task_evals[tid] = {
                "task_id": tid,
                "name": task_def.name,
                "description": task_def.description,  # include task description for LLM
                "difficulty": task_def.difficulty,
                "raw_metrics": metrics_for_task,  # all metrics
                "primary_eval": primary_eval,  # score computed on the primary metric
            }

            weighted_scores.append(float(primary_eval["weighted_score"]))
            weights_sum += primary_eval["difficulty_weight"]

        # aggregate difficulty-weighted scores
        if not weighted_scores or weights_sum <= 0:
            raise ValueError("No valid task scores to aggregate in evaluation.")
        overall_weighted_score = sum(weighted_scores) / weights_sum  # in [0,1]

        # obtain final score in [0,10]
        final_score_0_to_10 = max(0.0, min(10.0, 10.0 * overall_weighted_score))

        evaluation_summary: dict[str, object] = {
            "task_type": task_type,
            "primary_metric": PRIMARY_METRIC[task_type],
            "normalization_formula": "score = 1 / (1 + a * loss^b)",  # LLM needs this
            "metric_normalization_params": METRIC_NORMALIZATION,
            "difficulty_weights": DIFFICULTY_WEIGHTS,
            "num_tasks": len(tasks_by_id),
            "per_task": per_task_evals,
            "overall_weighted_score_0_to_1": overall_weighted_score,
            "final_score_0_to_10": final_score_0_to_10,
        }

        # generate LLM-based feedback
        feedback: str | None = None
        try:
            feedback = await self._generate_feedback(task_type, evaluation_summary)
            if feedback:
                evaluation_summary["feedback"] = feedback
        except Exception as e:
            logger.warning("LLM feedback generation failed: %s", e)

        logger.info(
            f"Evaluation complete for task_type='{task_type}'. "
            f"Evaluation Summary: {json.dumps(evaluation_summary, indent=2)}"
        )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Evaluation complete for task_type='{task_type}'. "
                f"Evaluation Summary: \n {json.dumps(evaluation_summary, indent=2)}",
                context_id=updater.context_id,
            ),
        )

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
        if cfg is None:
            a = 1.0
            b = 1.0
        else:
            a = float(cfg.get("a", 1.0))
            b = float(cfg.get("b", 1.0))

        # s = 1 / (1 + a * value^b) in (0,1]
        s = 1.0 / (1.0 + a * (raw_value**b))
        return max(0.0, min(1.0, s))

    def _compute_primary_metric_score(
        self,
        task_type: str,
        difficulty: str,
        metrics: dict[str, float],
    ) -> dict:
        """
        Compute normalized + weighted score from the PRIMARY_METRIC.
        Other metrics are not normalized for scoring, but are included
        in the evaluation summary for LLM feedback.
        """

        primary = PRIMARY_METRIC[task_type]
        raw_value = float(metrics[primary])

        # normalize using METRIC_NORMALIZATION
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

    async def _generate_feedback(self, task_type: str, summary: dict) -> str | None:
        if not USE_LLM_FEEDBACK:
            return None

        primary = summary["primary_metric"]
        score = summary["final_score_0_to_10"]

        prompt = f"""
    You are an expert evaluator for time-series machine learning models.

    The task_type is: '{task_type}'.

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
    6. Reference secondary metrics (MAE, QuantileLoss, VAR, CrossCorr)
       to support your reasoning.

    ENSURE that your feedback if **complete**
    """

        response = await acompletion(
            model="gpt-4o-mini",  # "bedrock/amazon.titan-text-lite-v1"
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.2,
        )

        msg = response["choices"][0]["message"]["content"]
        return msg if isinstance(msg, str) else json.dumps(msg)

    # legacy helper: average all the normalized scores
    def _compute_task_score(
        self,
        task_type: str,
        difficulty: str,
        metric_values: dict[str, float],
    ) -> dict[str, float | dict[str, float]]:
        """
        Given raw metric values for a single task, compute:

        (1) per-metric normalized scores in (0,1]
        (2) averaged score in (0,1]
        (3) difficulty-weighted score in (0, +inf) (will be scaled later)
        """

        metric_names = METRICS_BY_TYPE[task_type]

        # normalize each metric
        normalized_scores: dict[str, float] = {}
        for m in metric_names:
            raw = float(metric_values[m])
            normalized_scores[m] = self._normalize_metric(m, raw)

        # average per-metric scores to overall score
        if not normalized_scores:
            raise ValueError("No metrics to aggregate for task scoring.")
        avg_score = sum(normalized_scores.values()) / len(normalized_scores)

        # difficulty weight
        weight = DIFFICULTY_WEIGHTS.get(difficulty, 1.0)
        weighted_score = avg_score * weight

        return {
            "avg_score": avg_score,
            "weighted_score": weighted_score,
            "per_metric_scores": normalized_scores,
        }

    def _is_assignment_request(self, request: EvalRequest) -> bool:
        """
        Assignment request: config has task_type, no results.
        """

        config = request.config or {}
        return "task_type" in config and "results" not in config

    def _is_evaluation_request(self, request: EvalRequest) -> bool:
        """
        Evaluation request: has task_type and results (structure is checked separately).
        """

        config = request.config or {}
        return "task_type" in config and "results" in config

    def _validate_evaluation_config(self, config: dict) -> tuple[bool, str]:
        """
        Structure validation for evaluation request.

        Expected:
        {
            "task_type": <str in ALLOWED_TASK_TYPES>,
            "results": [
                {
                    "task_id": <str>,
                    "metrics": {
                        <metric_name>: <float>,
                        ...
                    }
                },
                ...
            ]
        }
        """
        task_type = config.get("task_type")
        if task_type not in ALLOWED_TASK_TYPES:
            return (
                False,
                f"'task_type' must be one of {ALLOWED_TASK_TYPES}, got {task_type!r}",
            )

        expected_metrics = METRICS_BY_TYPE.get(task_type)
        if not expected_metrics:
            return False, f"No expected metrics configured for task_type='{task_type}'."

        results = config.get("results")
        if not isinstance(results, list) or not results:
            return False, "'results' must be a non-empty list."

        seen_task_ids: set[str] = set()

        for idx, item in enumerate(results):
            if not isinstance(item, dict):
                return False, f"results[{idx}] must be a dict."

            tid = item.get("task_id")
            if not isinstance(tid, str) or not tid.strip():
                return False, f"results[{idx}].task_id must be a non-empty string."

            if tid in seen_task_ids:
                return False, f"Duplicate task_id in results: {tid}"
            seen_task_ids.add(tid)

            metrics = item.get("metrics")
            if not isinstance(metrics, dict) or not metrics:
                return False, f"results[{idx}].metrics must be a non-empty dict."

            # Check required metrics are present and numeric
            for m in expected_metrics:
                if m not in metrics:
                    return False, (
                        f"results[{idx}] for task_id={tid} is missing metric '{m}'. "
                        f"Expected metrics: {expected_metrics}"
                    )
                v = metrics[m]
                if not isinstance(v, (int, float)):
                    return False, (
                        f"Metric '{m}' for task_id={tid} must be numeric, "
                        f"got {type(v).__name__}."
                    )
                if v < 0:
                    return False, f"Metric '{m}' must be non-negative."

        # Check task_ids exist in TaskBank for that task_type
        registered_tasks = self.task_bank.get_tasks_by_type(task_type)
        registered_ids = {t.task_id for t in registered_tasks}
        if not registered_ids:
            return False, f"No tasks registered for task_type='{task_type}'."

        unknown_ids = seen_task_ids - registered_ids
        if unknown_ids:
            return (
                False,
                f"Unknown task_ids for task_type='{task_type}': {sorted(unknown_ids)}",
            )

        # Check all task_ids has evaluation results
        missing_ids = registered_ids - seen_task_ids
        if missing_ids:
            return False, (
                "Evaluation results missing some required tasks for task_type="
                f"'{task_type}': {sorted(missing_ids)}"
            )

        return True, "Valid evaluation request."

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        """
        Validate EvalRequest for both task assignment and evaluation.

        - task assignment: task_type in config, no results.
        - evaluation: task_type + results with proper metrics.
        """

        config = request.config or {}

        # task assignment request
        if self._is_assignment_request(request):
            task_type = config.get("task_type")
            if not isinstance(task_type, str) or not task_type.strip():
                return False, "'task_type' must be a non-empty string."

            if task_type not in ALLOWED_TASK_TYPES:
                return False, f"'task_type' must be one of {ALLOWED_TASK_TYPES}"

            tasks = self.task_bank.get_tasks_by_type(task_type)
            if not tasks:
                return False, f"No tasks available for task_type='{task_type}'."

            return True, "Valid task assignment request."

        # scoring request
        if self._is_evaluation_request(request):
            ok, msg = self._validate_evaluation_config(config)
            return ok, msg

        return False, (
            "Invalid request: must contain either "
            "'task_type' in config for task assignment, or "
            "'task_type' + 'results for evaluation."
        )


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

    s3_bucket_name = os.getenv("S3_BUCKET")

    file_dir = Path(__file__).resolve().parent
    proj_dir = file_dir.parents[2]

    tasks_json_path = (proj_dir / "data/tasks.json").resolve()

    async with agent_url_cm as agent_url:
        task_bank = TaskBank(
            s3_bucket=s3_bucket_name,
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
