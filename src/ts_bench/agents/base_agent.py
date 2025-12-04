from abc import abstractmethod

from a2a.server.tasks import TaskUpdater

from ts_bench.types import EvalRequest


class GreenAgent:
    @abstractmethod
    async def run_eval(self, request: EvalRequest, updater: TaskUpdater) -> None:
        pass

    @abstractmethod
    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        pass
