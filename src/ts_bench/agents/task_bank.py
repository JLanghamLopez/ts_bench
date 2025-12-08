import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    profile_name = os.getenv("AWS_PROFILE")
    region_name = os.getenv("AWS_REGION", "us-east-1")

    session_kwargs: Dict[str, Any] = {}
    if profile_name:
        session_kwargs["profile_name"] = profile_name
    if region_name:
        session_kwargs["region_name"] = region_name

    session = boto3.Session(**session_kwargs)
    s3 = session.client("s3")

    logger.info("Successfully initialized boto3 client.")

except (BotoCoreError, ClientError) as e:
    logger.error(f"Error initializing boto3 client: {e}")
    raise


class TaskDefinition(BaseModel):
    task_id: str
    name: str
    description: str
    task_type: str  # "time-series-generation", "time-series-forecasting"
    difficulty: str  # "easy", "medium", "hard"
    data_s3_key: str
    eval_fn_s3_key: str


class TaskBank:
    def __init__(
        self,
        s3_bucket,
        tasks_json_path: str = "../data/tasks.json",
    ):
        self.s3_bucket = s3_bucket

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

    def get_presigned_url(
        self, s3_key: str, s3_bucket: Optional[str] = None, expiration: int = 3600
    ) -> str:
        """Return a presigned URL (string) for a given S3 key."""

        s3_bucket = s3_bucket or self.s3_bucket
        try:
            response = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": s3_bucket, "Key": s3_key},
                ExpiresIn=expiration,
            )
            return response
        except ClientError as e:
            logger.error(f"Could not generate presigned URL for {s3_key}: {e}")
            raise


if __name__ == "__main__":
    s3_bucket_name = os.getenv("S3_BUCKET")

    file_dir = Path(__file__).resolve().parent
    proj_dir = file_dir.parents[2]

    tasks_json_path = (proj_dir / "data/tasks.json").resolve()

    task_bank = TaskBank(
        s3_bucket=s3_bucket_name,
        tasks_json_path=str(tasks_json_path),
    )
    logger.info("TaskBank initialised with %d tasks.", len(task_bank._tasks_by_id))

    task = task_bank.get_task("stock_forecast")
    print("DATA URL:", task_bank.get_presigned_url(task.data_s3_key))
    print("EVAL URL:", task_bank.get_presigned_url(task.eval_fn_s3_key))
