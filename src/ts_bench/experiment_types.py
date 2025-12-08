from typing import Any

from pydantic import BaseModel, HttpUrl


class EvalRequest(BaseModel):
    participant: HttpUrl
    config: dict[str, Any]


class EvalResult(BaseModel):
    score: float
    feedback: str


class TaskAssignment(BaseModel):
    task_id: str
    name: str
    description: str
    task_type: str
    difficulty: str
    data_url: HttpUrl
    eval_fn_url: HttpUrl
