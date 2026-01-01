FROM python:3.11-buster

RUN pip install poetry==2.1.4

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY ./pyproject.toml pyproject.toml
COPY ./poetry.lock poetry.lock
COPY ./src/ src
COPY ./README.md README.md

RUN poetry install --without dev && rm -rf $POETRY_CACHE_DIR

EXPOSE 8080

CMD ["poetry", "run", "python", "-m", "ts_bench.agents.ts_task_agent.agent", "--host", "0.0.0.0", "--port", "8080"]
