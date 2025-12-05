from a2a.types import AgentCapabilities, AgentCard


def ts_task_agent_card(url: str) -> AgentCard:
    return AgentCard(
        name="TS-Bench Agent",
        description="Time series ML problem task generation",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[],
        supports_authenticated_extended_card=True,
    )
