import os
from a2a.types import AgentCapabilities, AgentCard

public_agent_card = AgentCard(
    name="TS-Bench Agent",
    description="Time series ML problem task generation and evaluation for commodity, crypto, and FX markets. Supports forecasting and generative modeling tasks.",
    url=os.getenv("AGENT_URL", "http://localhost:9999/"),
    version="1.0.0",
    default_input_modes=["text"],
    default_output_modes=["text"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[
        "time-series-forecasting",
        "generative-modeling",
        "commodity-markets",
        "cryptocurrency-markets",
        "fx-markets",
        "task-assignment",
        "evaluation"
    ],
    supports_authenticated_extended_card=True,
    provider="my-org",
    supported_languages=["en"],
)
