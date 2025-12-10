from typing import Any

from pydantic import BaseModel, HttpUrl


class EvalRequest(BaseModel):
    participant: HttpUrl
    config: dict[str, Any]


class EvalResult(BaseModel):
    score: float
    feedback: str
