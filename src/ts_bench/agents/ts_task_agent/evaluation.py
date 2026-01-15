import logging

from pydantic import BaseModel

from ts_bench.task_bank import TaskDefinition, TaskDifficulty, TaskType

logger = logging.getLogger(__name__)


PRIMARY_METRIC: dict[TaskType, str] = {
    TaskType.TIME_SERIES_FORECASTING: "rmse",
    TaskType.TIME_SERIES_GENERATION: "histloss",
}

DIFFICULTY_WEIGHTS: dict[TaskDifficulty, float] = {
    TaskDifficulty.EASY: 1.0,
    TaskDifficulty.INTERMEDIATE: 3.0,
    TaskDifficulty.ADVANCED: 5.0,
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
    score: float


class EvalSummary(BaseModel):
    task_type: TaskType
    primary_metric: str
    num_tasks: int
    per_task: list[TaskResult]
    overall_weighted_score: float


def failed_result(task: TaskDefinition) -> TaskResult:
    if task.task_type is TaskType.TIME_SERIES_FORECASTING:
        null_metrics = {"rmse": 1000000.0, "mae": 1000000.0, "mape": 10000000.0}
    else:
        null_metrics = {"histloss": 10000000.0, "auto_corr": 0.0, "cross_corr": 0.0}

    return TaskResult(
        task_id=task.task_id,
        name=task.name,
        description=task.description,
        difficulty=task.difficulty,
        raw_metrics=null_metrics,
        score=0.0,
    )


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


def _compute_score(
    metrics: dict[str, float],
) -> float:
    """
    Compute average normalized score
    """

    if not metrics:
        raise ValueError("metrics must be a non-empty dict")

    scores = [_normalize_metric(m, v) for m, v in metrics.items()]
    return float(sum(scores) / len(scores))


async def aggregate_scores(
    task_type: TaskType, results: list[TaskResult]
) -> EvalSummary:
    # Aggregate scores
    weighted_scores = [v.score * DIFFICULTY_WEIGHTS[v.difficulty] for v in results]
    weights_sum = sum([DIFFICULTY_WEIGHTS[v.difficulty] for v in results])

    overall_weighted_score = (
        sum(weighted_scores) / weights_sum if weights_sum > 0 else 0.0
    )

    return EvalSummary(
        task_type=task_type.value,
        primary_metric=PRIMARY_METRIC[task_type],
        num_tasks=len(results),
        per_task=results,
        overall_weighted_score=overall_weighted_score,
    )
