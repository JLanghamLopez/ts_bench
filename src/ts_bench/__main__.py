from pathlib import Path

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from data.task_bank import TaskBank
from ts_bench.agents.agent_card import public_agent_card
from ts_bench.agents.ts_task_agent import TSTaskAgent
from ts_bench.executor import TSBenchExecutor

if __name__ == "__main__":
    file_dir = Path(__file__).resolve().parent
    proj_dir = file_dir.parents[1]

    tasks_json_path = (proj_dir / "data/tasks.json").resolve()

    task_bank = TaskBank(
        tasks_json_path=str(tasks_json_path),
    )
    green_agent = TSTaskAgent(task_bank=task_bank)

    request_handler = DefaultRequestHandler(
        agent_executor=TSBenchExecutor(agent=green_agent),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(server.build(), host="0.0.0.0", port=9999)
