import os
from dotenv import load_dotenv
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from ts_bench.agent_card import public_agent_card
from ts_bench.executor import TSBenchExecutor
from ts_bench.green_agent import TimeSeriesGreenAgent
from ts_bench.task_bank import TaskBank
from pathlib import Path

if __name__ == "__main__":
    file_dir = Path(__file__).resolve().parent
    proj_dir = file_dir.parents[1] 

    db_path = (proj_dir / "lancedb_tasks").resolve() 
    db_path.mkdir(parents=True, exist_ok=True)

    tasks_json_path = (proj_dir / "data/tasks.json").resolve() 

    # Configuration from environment variables
    load_dotenv()
    s3_bucket_name = os.getenv("S3_BUCKET", "competition-bucket-S3")
    embedding_model = os.getenv("EMBEDDING_MODEL", "hkunlp/instructor-large")

    task_bank = TaskBank(
        s3_bucket_name=s3_bucket_name,
        db_path=str(db_path),
        tasks_json_path=str(tasks_json_path),
        embedding_model=embedding_model
    )
    green_agent = TimeSeriesGreenAgent(task_bank=task_bank)

    request_handler = DefaultRequestHandler(
        agent_executor=TSBenchExecutor(agent=green_agent),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "9999"))

    uvicorn.run(server.build(), host=host, port=port)
