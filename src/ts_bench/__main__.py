import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from ts_bench.agent_card import public_agent_card
from ts_bench.executor import TSBenchExecutor

if __name__ == "__main__":
    request_handler = DefaultRequestHandler(
        agent_executor=TSBenchExecutor(),
        task_store=InMemoryTaskStore(),
    )

    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )

    uvicorn.run(server.build(), host="0.0.0.0", port=9999)
