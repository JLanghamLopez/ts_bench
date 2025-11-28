import logging
import re

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue

logger = logging.getLogger(__name__)


def parse_tags(str_with_tags: str) -> dict[str, str]:
    """
    The target str contains tags in the format of <tag_name> ... </tag_name>,
    parse them out and return a dict
    """
    tags = re.findall(r"<(.*?)>(.*?)</\1>", str_with_tags, re.DOTALL)
    return {tag: content.strip() for tag, content in tags}


class TSBenchExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        logger.info("Received a task, parsing...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        white_agent_url = tags["white_agent_url"]

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass
