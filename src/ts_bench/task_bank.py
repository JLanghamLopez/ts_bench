from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    TIME_SERIES_GENERATION = "time-series-generation"
    TIME_SERIES_FORECASTING = "time-series-forecasting"


class TaskDifficulty(str, Enum):
    EASY = "Easy"
    INTERMEDIATE = "Intermediate"
    ADVANCED = "Advanced"


class TaskDefinition(BaseModel):
    task_id: str
    name: str
    task_type: TaskType
    difficulty: TaskDifficulty
    description: str
    url: str


class TaskBank:
    def __init__(self, tasks_json_path: str):
        # Build task base
        self._tasks_by_id: Dict[str, TaskDefinition] = {}
        self._tasks_by_type: Dict[TaskType, List[TaskDefinition]] = {}

        self._load_tasks_from_json(tasks_json_path)

    def _load_tasks_from_json(self, tasks_json_path: str):
        path = Path(tasks_json_path)

        if not path.exists():
            msg = f"Tasks JSON file not found at {tasks_json_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            with path.open("r", encoding="utf-8") as f:
                raw_tasks = json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse tasks JSON (%s): %s", tasks_json_path, e)
            raise e

        if not raw_tasks:
            msg = "No tasks found in the JSON file."
            logger.info(msg)
            raise ValueError(msg)

        loaded = 0

        for raw_task in raw_tasks:
            try:
                task = TaskDefinition(**raw_task)
            except Exception as e:
                logger.error("Failed to parse task from JSON entry %s: %s", raw_task, e)
                continue

            # store by id
            if task.task_id in self._tasks_by_id:
                logger.warning(
                    "Duplicate task_id '%s' found in JSON. Overwriting previous entry.",
                    task.task_id,
                )

            self._tasks_by_id[task.task_id] = task

            # store by type (Enum key)
            self._tasks_by_type.setdefault(task.task_type, []).append(task)
            loaded += 1

        logger.info(
            "Loaded %d tasks from JSON (%d task types).",
            loaded,
            len(self._tasks_by_type),
        )

    @staticmethod
    def _to_task_type(task_type: TaskType | str) -> TaskType:
        if isinstance(task_type, TaskType):
            return task_type
        if not isinstance(task_type, str):
            raise TypeError(f"task_type must be str or TaskType, got {type(task_type)}")
        try:
            return TaskType(task_type)
        except ValueError as e:
            raise ValueError(
                f"Unknown task_type '{task_type}'. "
                f"Must be one of {[t.value for t in TaskType]}"
            ) from e

    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        task = self._tasks_by_id.get(task_id)
        if not task:
            logger.warning("No task found with task_id='%s'.", task_id)
        return task

    def get_tasks_by_type(self, task_type: TaskType | str) -> List[TaskDefinition]:
        """
        Return all tasks of a given type. Accepts either TaskType Enum or string value.
        """
        try:
            ttype = self._to_task_type(task_type)
        except ValueError as e:
            logger.warning("get_tasks_by_type called with invalid task_type: %s", e)
            return []

        tasks = self._tasks_by_type.get(ttype, [])
        if not tasks:
            logger.warning("No tasks found for task_type=%s", ttype.value)
        return list(tasks)

    def get_all_task_ids_by_type(self, task_type: TaskType | str) -> List[str]:
        tasks = self.get_tasks_by_type(task_type)
        return [task.task_id for task in tasks]

    def get_single_task_by_type(
        self,
        task_type: TaskType | str,
        index: int = 0,
    ) -> Optional[TaskDefinition]:
        """
        Return a single task for a given type.
        """
        tasks = self.get_tasks_by_type(task_type)
        if not tasks:
            return None

        if index < 0 or index >= len(tasks):
            logger.warning(
                "Index %d out of range for task_type='%s' (num_tasks=%d).",
                index,
                self._to_task_type(task_type).value,
                len(tasks),
            )
            return None

        return tasks[index]

    def get_url(self, task_id: str) -> Optional[str]:
        task = self._tasks_by_id.get(task_id)
        if not task:
            logger.warning(
                "No task found with task_id='%s' when requesting url.", task_id
            )
            return None
        return task.url

    def all_tasks(self) -> Iterable[TaskDefinition]:
        return self._tasks_by_id.values()
