import os
import json
from dotenv import load_dotenv
import boto3
from botocore.exceptions import ClientError, BotoCoreError
import lancedb
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, HttpUrl
from lancedb.pydantic import LanceModel, Vector
from sentence_transformers import SentenceTransformer
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_SIZE = 768 # Based on 'hkunlp/instructor-large' or 'intfloat/e5-base-v2'

class TaskDefinition(LanceModel):
    task_id: str
    name: str
    description: str
    vector: Vector(EMBEDDING_SIZE) # type: ignore
    task_type: str # "forecasting", "generation"
    difficulty: str #"easy", "medium", "hard"
    data_s3_key: str
    eval_fn_s3_key: str

class TaskBank:
    def __init__(
        self,
        s3_bucket_name: str = "competition-bucket-s3", # Added s3_bucket_name parameter
        db_path: str = "lancedb_tasks",
        tasks_json_path: str = "00/00_task/ts_bench/data/tasks.json", # default path
        embedding_model: str = "hkunlp/instructor-large",
    ):

        # Initialize client
        self.s3_bucket_name = s3_bucket_name
        self.s3_client = self._init_s3_client()
        
        # Initialize LanceDB
        self.db = lancedb.connect(db_path)
        self.embedder = SentenceTransformer(embedding_model)
        self.table = self.db.create_table("tasks", schema=TaskDefinition, exist_ok=True)
        logger.info(f"LanceDB connected to {db_path}, table 'tasks' ready.")

        self._load_tasks_from_json(tasks_json_path)
    
    @staticmethod
    def _init_s3_client():
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
            return s3

        except (BotoCoreError, ClientError) as e:
            logger.error(f"Error initializing boto3 client: {e}")
            raise

    def _load_tasks_from_json(self, tasks_json_path: str):
        if not Path(tasks_json_path).exists():
            logger.error(f"Tasks JSON file not found at {tasks_json_path}")
            return

        with open(tasks_json_path, "r", encoding="utf-8") as f:
            raw_tasks = json.load(f)
        
        if not raw_tasks:
            logger.info("No tasks found in the JSON file.")
            return

        descriptions = [t["description"] for t in raw_tasks]
        embeddings = self.embedder.encode(
            descriptions, normalize_embeddings=True
        ).tolist()

        task_entries = []
        for raw_task, vec in zip(raw_tasks, embeddings):
            task_entries.append(
                TaskDefinition(
                    task_id=raw_task["task_id"],
                    name=raw_task["name"],
                    description=raw_task["description"],
                    vector=vec,
                    task_type=raw_task["task_type"],
                    difficulty=raw_task["difficulty"],
                    data_s3_key=raw_task["data_s3_key"],
                    eval_fn_s3_key=raw_task["eval_fn_s3_key"],
                )
            )
        
       
        if self.table.count_rows() == 0:
            self.table.add(task_entries)
            logger.info(f"Loaded {len(task_entries)} tasks into LanceDB.")
        else:
             logger.info(f"Using existing {self.table.count_rows()} tasks from LanceDB.")

    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        # where clause for exact match on task_id
        results = self.table.where(f"task_id = '{task_id}'").limit(1).to_pydantic(TaskDefinition)
        return results[0] if results else None

    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> HttpUrl:
        try:
            response = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.s3_bucket_name, "Key": s3_key},
                ExpiresIn=expiration,
            )
            return HttpUrl(response)
        except ClientError as e:
            logger.error(f"Could not generate presigned URL for {s3_key}: {e}")
            raise

    def search_tasks(self, query: str, k: int = 10) -> List[TaskDefinition]:
        embedding = self.embedder.encode([query], normalize_embeddings=True)[0]
        retrieve_results = self.table.search(embedding).limit(k).to_pydantic(TaskDefinition)

        if not retrieve_results:
            logger.info(f"No matching tasks found for query: '{query}'")
            return []

        logger.info(f"Retrieved {len(retrieve_results)} tasks for query: '{query}'")
        return retrieve_results

if __name__ == "__main__":
    load_dotenv()
    
    file_dir = Path(__file__).resolve().parent
    proj_dir = file_dir.parents[1] 

    db_path = (proj_dir / "lancedb_tasks").resolve() 
    db_path.mkdir(parents=True, exist_ok=True)

    tasks_json_path = (proj_dir / "data/tasks.json").resolve() 
    
    task_bank = TaskBank(
        s3_bucket_name="competition-bucket-s3, 
        db_path=str(db_path),
        tasks_json_path=str(tasks_json_path),
        embedding_model="hkunlp/instructor-large"
    )
    logger.info(f"Total tasks in LanceDB: {task_bank.table.count_rows()}")
