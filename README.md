# TS-Bench

## Overview

TS-Bench is a time-series benchmark evaluation system using an Agent-to-Agent (A2A)
architecture, and developed for the [AgentBeats](https://agentbeats.dev/) platform.

The **green agent**
([`TSTaskAgent`](https://github.com/JLanghamLopez/ts_bench/blob/main/src/ts_bench/agents/ts_task_agent/agent.py))
manages task distribution and evaluation for time-series forecasting and generation tasks.

## See also

- AgentBeats page: https://agentbeats.dev/JLanghamLopez/ts-bench
- Leaderboard repo: https://agentbeats.dev/JLanghamLopez/ts-bench

## Green Agent Workflow

The green agent roughly follows these steps:

**Trigger**: Receive `EvalRequest` with `task_type` and participant details.

**Process**:
1. Validate task type (`time-series-forecasting` or `time-series-generation`)
2. Retrieve all tasks of that type from the `TaskBank`
3. For each task:
  - Generate a text task instruction
  - Send task instructions to the purple agent
  - Check the format of the prediction data returned by the purple agent
  - Apply test metrics to the prediction data, and score the task
4. Once all the tasks are complete, aggregate the scores with weightings
5. Complete the assessment

## Metrics

The data produced by the participating agent is scored against hold-out
test data using the following metrics:

### Forecasting Tasks

- RMSE: Root mean squared error measuring overall forecast accuracy.
- MAE: Mean absolute error providing a stable magnitude-based measure.
- MAPE: Mean absolute percentage error offering a scale-normalised comparison.

### Generation Tasks

- HistogramMetric: Measures the similarity between real and generated time series by comparing their empirical marginal distributions.
- CrossCorrelationMetric: Evaluates whether dependence structures across different dimensions of the time series are preserved.
- AutoCorrelationMetric: Assesses temporal dependence within individual time series components.

## Example Participant (Purple) Agents

This repo also includes two example agents participant agents:

- [BaselineExecutorExecutor](https://github.com/JLanghamLopez/ts_bench/blob/main/src/ts_bench/agents/benchmark_agent.py)
  a basic agent that downloads the task datasets, and prompts an LLM to generate
  a Python module that generates the relevant task output.
- [PresetExecutorExecutor](https://github.com/JLanghamLopez/ts_bench/blob/main/src/ts_bench/agents/deterministic_agent.py)
  an agent that generates random but deterministic outputs, for testing reproducibility.

## Agent Customisation and Building

Tasks can be modified or added by modify
[`data/tasks.yaml`](https://github.com/JLanghamLopez/ts_bench/blob/main/data/tasks.yaml)
and then building the agent with

```commandline
docker build -f dockerfiles/Dockerfile.green .
```

## Developers

### Installation

Dependencies can be installed with [poetry](https://python-poetry.org/) by running

```commandline
poetry install --with benchmark
```

which will create a virtual environment in the repo at `.venv` which can
be activated with

```commandline
source .venv/bin/activate
```

On Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### Code Checks

Linting/formatting can be run using pre-commit

```commandline
pre-commit run --all-files
```
