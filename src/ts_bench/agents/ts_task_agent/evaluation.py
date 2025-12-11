import json
import logging

from litellm import acompletion
from pydantic import BaseModel

from data.task_bank import TaskDefinition, TaskDifficulty, TaskType

logger = logging.getLogger(__name__)

USE_LLM_FEEDBACK = False

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


class TaskResult(BaseModel):
    task_id: str
    name: str
    description: str
    difficulty: TaskDifficulty
    raw_metrics: dict[str, float]
    primary_eval: float
    prediction_path: str


class EvalSummary(BaseModel):
    task_type: TaskType
    primary_metric: str
    num_tasks: int
    per_task: list[TaskResult]
    overall_weighted_score_0_to_1: float
    final_score_0_to_10: float
    feedback: str


def failed_result(predictions_path: str, task: TaskDefinition) -> TaskResult:
    # TODO: Check null results make sense for metrics
    if task.task_type is TaskType.TIME_SERIES_FORECASTING:
        null_metrics = {"rmse": 1000000.0, "mae": 1000000.0, "mape": 10000000.0}
        primary_eval = null_metrics["rmse"]
    else:
        null_metrics = {"histloss": 10000000.0, "auto_corr": 0.0, "cross_corr": 0.0}
        primary_eval = null_metrics["histloss"]

    return TaskResult(
        task_id=task.task_id,
        name=task.name,
        description=task.description,
        difficulty=task.difficulty,
        raw_metrics=null_metrics,
        primary_eval=primary_eval,
        prediction_path=predictions_path,
    )


async def evaluate_predictions(
    predictions_path: str, assignment: TaskDefinition
) -> TaskResult:
    """
    Run the correct_fn evaluation for the given task_type.
    This is a placeholder that will call the appropriate evaluation function.

    - Load predictions CSV from predictions[task_id]
    - Call task-specific eval_fn (TODO) to compute raw_metrics
    - Compute primary metric normalized score and difficulty-weighted score
    """
    task_type = assignment.task_type
    logger.info("Running evaluation for task_type='%s'.", assignment.task_type.value)
    # primary_metric = PRIMARY_METRIC[task_type]

    logger.info(
        "Evaluating task_id='%s', difficulty='%s', prediction_path='%s'",
        assignment.task_id,
        assignment.difficulty.value,
        predictions_path or "<missing>",
    )

    # TODO: call correct eval_fn.py
    # raw_metrics = await self._run_single_task_eval(task, pred_path, metric_names)
    # Placeholder
    if task_type is TaskType.TIME_SERIES_FORECASTING:
        raw_metrics = {"rmse": 0.5, "mae": 0.4, "mape": 0.3}
    else:
        raw_metrics = {"histloss": 0.6, "auto_corr": 0.5, "cross_corr": 0.4}

    primary_eval = _compute_primary_metric_score(
        task_type=task_type,
        difficulty=assignment.task.difficulty,
        metrics=raw_metrics,
    )

    result = TaskResult(
        task_id=assignment.task_id,
        name=assignment.name,
        description=assignment.description,
        difficulty=assignment.difficulty,
        raw_metrics=raw_metrics,
        primary_eval=primary_eval,
        prediction_path=predictions_path,
    )

    return result


def _normalize_metric(
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
    task_type: TaskType,
    difficulty: TaskDifficulty,
    metrics: dict[str, float],
) -> dict:
    """
    Compute normalized + weighted score from the PRIMARY_METRIC.
    """
    primary = PRIMARY_METRIC[task_type]
    raw_value = float(metrics[primary])

    normalized_score = _normalize_metric(primary, raw_value)
    weight = DIFFICULTY_WEIGHTS.get(difficulty, 1.0)
    weighted = normalized_score * weight

    return {
        "primary_metric": primary,
        "raw_loss": raw_value,
        "normalized_score": normalized_score,
        "difficulty_weight": weight,
        "weighted_score": weighted,
    }


async def aggregate_scores(
    task_type: TaskType, results: list[TaskResult]
) -> EvalSummary:
    # Aggregate scores
    weighted_scores = [v.primary_eval["weighted_score"] for v in results]
    weights_sum = sum([v.primary_eval["difficulty_weight"] for v in results])

    overall_weighted_score = (
        sum(weighted_scores) / weights_sum if weights_sum > 0 else 0.0
    )
    final_score = max(0.0, min(10.0, 10.0 * overall_weighted_score))

    if USE_LLM_FEEDBACK:
        evaluation_summary = {
            "task_type": task_type.value,
            "primary_metric": PRIMARY_METRIC[task_type],
            "num_tasks": len(results),
            "per_task": results,
            "overall_weighted_score_0_to_1": overall_weighted_score,
            "final_score_0_to_10": final_score,
            # for LLM feedback prompt
            "metric_normalization_params": METRIC_NORMALIZATION,
            "difficulty_weights": {d.value: w for d, w in DIFFICULTY_WEIGHTS.items()},
        }

        try:
            feedback = await _generate_feedback(task_type, evaluation_summary)

        except Exception as e:
            logger.warning("LLM feedback generation failed: %s", e)
            raise e

    else:
        feedback = "No feedback provided"

    return EvalSummary(
        task_type=task_type.value,
        primary_metric=PRIMARY_METRIC[task_type],
        num_tasks=len(results),
        per_task=results,
        overall_weighted_score_0_to_1=overall_weighted_score,
        final_score_0_to_10=final_score,
        feedback=feedback,
    )


async def _generate_feedback(task_type: TaskType, summary: dict) -> str | None:
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
