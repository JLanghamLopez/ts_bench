from a2a.types import AgentCapabilities, AgentCard, AgentSkill


def ts_task_agent_card(url: str) -> AgentCard:
    skills = [
        AgentSkill(
            id="ts-forecasting",
            name="time-series-forecasting",
            description="Forecasting tasks for multivariate and financial time series.",
            tags=["forecasting", "time-series", "ml"],
        ),
        AgentSkill(
            id="gen-modeling",
            name="time-series-generation",
            description="Generative modeling tasks for stochastic time series.",
            tags=["generation", "time-series", "ml"],
        ),
        AgentSkill(
            id="task-assign",
            name="task-assignment",
            description="Select and assign benchmark ML tasks.",
            tags=["assignment", "benchmark"],
        ),
        AgentSkill(
            id="task-eval",
            name="evaluation",
            description="Evaluate submitted predictions or generated samples.",
            tags=["evaluation", "metrics", "benchmark"],
        ),
    ]

    return AgentCard(
        name="TS-Bench Agent",
        description="Time series ML problem task generation",
        url=url,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=skills,
        supports_authenticated_extended_card=True,
    )
