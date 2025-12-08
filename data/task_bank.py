import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskDefinition(BaseModel):
    task_id: str
    name: str
    task_type: str  # "time-series-generation", "time-series-forecasting"
    difficulty: str  # "Easy", "Intermediate", "Advanced"
    url: str


class TaskBank:
    def __init__(
        self,
        tasks_json_path: str = "../data/tasks.json",
    ):
        # Build task base
        self._tasks_by_id: Dict[str, TaskDefinition] = {}
        self._tasks_by_type: Dict[str, List[TaskDefinition]] = {}

        self._load_tasks_from_json(tasks_json_path)

    def _load_tasks_from_json(self, tasks_json_path: str):
        path = Path(tasks_json_path)
        if not path.exists():
            logger.error(f"Tasks JSON file not found at {tasks_json_path}")
            return

        with path.open("r", encoding="utf-8") as f:
            raw_tasks = json.load(f)

        if not raw_tasks:
            logger.info("No tasks found in the JSON file.")
            return

        for raw_task in raw_tasks:
            task = TaskDefinition(**raw_task)
            self._tasks_by_id[task.task_id] = task
            self._tasks_by_type.setdefault(task.task_type, []).append(task)

        logger.info(
            "Loaded %d tasks from JSON (%d task types).",
            len(self._tasks_by_id),
            len(self._tasks_by_type),
        )

    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        return self._tasks_by_id.get(task_id)

    def get_tasks_by_type(self, task_type: str) -> List[TaskDefinition]:
        tasks = self._tasks_by_type.get(task_type, [])
        if not tasks:
            logger.warning("No tasks found for task_type=%s", task_type)
        return tasks

    def get_all_task_ids_by_type(self, task_type: str) -> List[str]:
        """Return a list of all task IDs for a given task type."""

        tasks = self.get_tasks_by_type(task_type)
        return [task.task_id for task in tasks]

    def get_url(self, task_id: str) -> str | None:
        task: TaskDefinition | None = self._tasks_by_id.get(task_id)
        if not task:
            return None
        return task.url


if __name__ == "__main__":
    file_dir = Path(__file__).resolve().parent
    proj_dir = file_dir.parents[0]

    tasks_json_path = (proj_dir / "data/tasks.json").resolve()

    task_bank = TaskBank(
        tasks_json_path=str(tasks_json_path),
    )
    logger.info("TaskBank initialised with %d tasks.", len(task_bank._tasks_by_id))

    task = task_bank.get_task("equity_forecasting")
    print("URL:", task_bank.get_url(task.task_id))
