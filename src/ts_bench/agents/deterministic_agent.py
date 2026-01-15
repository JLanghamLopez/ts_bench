import argparse
import asyncio
import contextlib
import json
import logging

import numpy as np
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


class PresetExecutorExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg_obj = context.get_user_input()

        logger.info(f"Participant agent received new task {context.task_id}")
        msg = context.message

        if msg is None:
            raise ServerError(error=InvalidParamsError(message="Missing message."))

        task = new_task(msg)
        await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                "Received assignment. Parsing tasks and generating solutions...",
                context_id=context.context_id,
            ),
        )

        msg_obj = json.loads(msg_obj)
        output_shape = msg_obj["task_specification"]["output_shape"]
        logger.info("Generating random predictions from fixed seed")
        rng = np.random.default_rng(101)
        preds = rng.normal(size=output_shape)

        payload = {"predictions": preds.tolist()}

        final_message = json.dumps(payload)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                final_message,
                context_id=context.context_id,
            ),
        )

        await event_queue.enqueue_event(
            new_agent_text_message(final_message, context_id=task.context_id)
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass


async def main():
    parser = argparse.ArgumentParser(description="Run benchmark participant agent.")
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
        executor = PresetExecutorExecutor()
        agent_card = AgentCard(
            name="Prest TS-Bench Purple Agent ",
            description="Time series agent that returns constant results",
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

        uvicorn_config = uvicorn.Config(server.build(), host=args.host, port=args.port)
        uvicorn_server = uvicorn.Server(uvicorn_config)
        await uvicorn_server.serve()


if __name__ == "__main__":
    asyncio.run(main())
