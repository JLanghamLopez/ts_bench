from a2a.types import AgentCapabilities, AgentCard

public_agent_card = AgentCard(
    name="TS-Bench Agent",
    description="Time series ML problem task generation",
    url="http://localhost:9999/",
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[],
    supports_authenticated_extended_card=True,
)
