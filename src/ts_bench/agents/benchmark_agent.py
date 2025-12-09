import argparse
import asyncio
import contextlib
import json
import logging

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    InvalidParamsError,
    TaskState,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaselineExecutorExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg_obj = context.get_user_input()

        logger.info(f"Participant agent received raw user_input object: {msg_obj}")

        msg = context.message

        logger.info(f"Participant agent received context.message: {msg}")
        if msg is None:
            raise ServerError(error=InvalidParamsError(message="Missing message."))
        task = new_task(msg)
        await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                "Participant received assignment. Generating deterministic predictions...",
                context_id=context.context_id,
            ),
        )

        predictions = {
            "commodity_forecasting": "/tmp/a.csv",
            "equity_forecasting": "/tmp/b.csv",
            "fx_forecasting": "/tmp/c.csv",
        }

        final_message = new_agent_text_message(
            json.dumps(predictions),
            context_id=task.context_id,
        )

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                json.dumps(predictions),
                context_id=context.context_id,
            ),
        )

        await event_queue.enqueue_event(final_message)

        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


async def main():
    parser = argparse.ArgumentParser(
        description="Run the time-series task generation agent."
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind the server"
    )
    parser.add_argument(
        "--port", type=int, default=9019, help="Port to bind the server"
    )
    parser.add_argument(
        "--card-url", type=str, help="External URL to provide in the agent card"
    )
    args = parser.parse_args()

    agent_url_cm = contextlib.nullcontext(
        args.card_url or f"http://{args.host}:{args.port}/"
    )

    async with agent_url_cm as agent_url:
        executor = BaselineExecutorExecutor()
        agent_card = AgentCard(
            name="TS-Bench Baseline Purple Agent",
            description="Baseline Time series ML problem solver",
            url=agent_url,
            version="1.0.0",
            default_input_modes=["text"],
            default_output_modes=["text"],
            capabilities=AgentCapabilities(streaming=False),
            skills=[],
            supports_authenticated_extended_card=True,
        )

        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )
        logger.info("Participant agent started")

        uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())
