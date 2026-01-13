from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List

import yaml
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
    eval_url: str
    data_url: str
    output_shape: list[int]


@dataclass
class Task:
    task_definition: TaskDefinition
    ground_truth_url: str


class TaskBank:
    def __init__(self, tasks_yaml_path: str):
        # Build task base
        self.loaded_tasks = 0
        self._tasks_by_id: Dict[str, Task] = {}
        self._tasks_by_type: Dict[TaskType, List[Task]] = {}
        self._load_tasks(tasks_yaml_path)

    def _load_tasks(self, tasks_yaml_path: str) -> None:
        path = Path(tasks_yaml_path)

        if not path.exists():
            msg = f"Tasks YAML file not found at {tasks_yaml_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            with path.open("r") as f:
                raw_tasks = yaml.safe_load(f)
        except Exception as e:
            logger.error("Failed to parse tasks YAML (%s): %s", tasks_yaml_path, e)
            raise e

        if not raw_tasks:
            msg = f"No tasks found in the YAML file {tasks_yaml_path}"
            logger.info(msg)
            raise ValueError(msg)

        for raw_task in raw_tasks:
            try:
                keys = TaskDefinition.model_fields
                task_def = TaskDefinition(**{k: raw_task[k] for k in keys.keys()})
                ground_truth_url = raw_task["ground_truth_url"]
                task = Task(task_definition=task_def, ground_truth_url=ground_truth_url)
            except Exception as e:
                logger.error("Failed to parse task from YAML entry %s: %s", raw_task, e)
                continue

            # store by id
            if task.task_definition.task_id in self._tasks_by_id:
                logger.warning(
                    "Duplicate task_id '%s' found in JSON. Overwriting previous entry.",
                    task.task_definition.task_id,
                )

            self._tasks_by_id[task.task_definition.task_id] = task

            # store by type (Enum key)
            self._tasks_by_type.setdefault(task.task_definition.task_type, []).append(
                task
            )
            self.loaded_tasks += 1

        logger.info(
            "Loaded %d tasks from YAML (%d task types).",
            self.loaded_tasks,
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

    def get_tasks_by_type(self, task_type: TaskType | str) -> List[Task]:
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
        return [task.task_definition.task_id for task in tasks]

    def all_tasks(self) -> Iterable[Task]:
        return self._tasks_by_id.values()
