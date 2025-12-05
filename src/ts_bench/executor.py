import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    InternalError,
    InvalidParamsError,
    Task,
    TaskState,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task, new_task_with_context
from a2a.utils.errors import ServerError
from pydantic import ValidationError

from ts_bench.green_agent import TimeSeriesGreenAgent
from ts_bench.types import EvalRequest, TaskAssignment, TaskRequest

logger = logging.getLogger(__name__)


class TSBenchExecutor(AgentExecutor):
    def __init__(self, agent: TimeSeriesGreenAgent) -> None:
        self.agent = agent

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        request_text = context.get_user_input()
        context_id = context.context_id

        msg = context.message
        if not msg:
            raise ServerError(error=InvalidParamsError(message="Missing message."))

        task = new_task(msg)
        await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue, task.id, context_id)

        try:
            # Try to parse as EvalRequest first
            try:
                req: EvalRequest = EvalRequest.model_validate_json(request_text)
                ok, validation_msg = self.agent.validate_request(req)
                if not ok:
                    raise ServerError(error=InvalidParamsError(message=validation_msg))

                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        f"Processing evaluation results.\n{req.model_dump_json()}",
                        context_id=context_id,
                    ),
                )
                await self.agent.run_eval(req, updater)
                await updater.complete()

            except ValidationError:
                # If not an EvalRequest, try to parse as TaskRequest
                try:
                    task_req: TaskRequest = TaskRequest(query=request_text) # Assume raw text is the query
                    ok, validation_msg = self.agent.validate_request(task_req)
                    if not ok:
                        raise ServerError(error=InvalidParamsError(message=validation_msg))

                    await updater.update_status(
                        TaskState.working,
                        new_agent_text_message(
                            f"Searching for a task matching your request: '{task_req.query}'",
                            context_id=context_id,
                        ),
                    )
                    task_assignment: TaskAssignment = await self.agent.handle_task_request(
                        task_req, context_id, updater
                    )

                    purple_agent_task = new_task_with_context(
                        f"Assigned task: {task_assignment.name}",
                        context_id=context_id,
                        details=task_assignment.model_dump(),
                    )
                    await event_queue.enqueue_event(purple_agent_task)
                    await updater.complete()

                except ValidationError as e:
                    raise ServerError(error=InvalidParamsError(message=f"Invalid request format: {e.json()}"))

        except ServerError:
            raise # Re-raise ServerError as is
        except Exception as e:
            logger.exception(f"Agent error: {e}")
            await updater.failed(
                new_agent_text_message(
                    f"Agent error: {e}", context_id=context_id
                )
            )
            raise ServerError(error=InternalError(message=str(e)))

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
