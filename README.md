# TS-Bench

## Overview

TS-Bench is a time-series benchmark evaluation system using an Agent-to-Agent (A2A)
architecture. The **green agent** ([ts_task_agent.py]) manages task distribution and
evaluation for time-series forecasting and generation tasks.

## Green Agent Workflow

The green agent (`TSTaskAgent`) roughly follows these steps:

**Trigger**: Receive `EvalRequest` with `task_type`

**Process**:
1. Validate task type ("time-series-forecasting" or "time-series-generation")
2. Retrieve all tasks of that type from `TaskBank`
3. For each task:
  - Generate a text task instruction
  - Send task instructions to the purple agent
  - Check the format of the prediction data returned by the purple agent
  - Apply test metrics to the prediction data, and score the task
4. Once all the tasks are complete, aggregate the scores with weightings
5. Complete the assessment

## Metrics

### Forecasting Tasks
- **Primary**: RMSE (Root Mean Square Error)
- Secondary: MAE, Quantile Loss

### Generation Tasks
- **Primary**: SigW1 (Signature Wasserstein-1)
- Secondary: Auto-correlation, Cross-correlation

## Developers

## Installation

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

## Running

Run an example scenario locally using

```commandline
python -m ts_bench.run_scenario scenario.toml
```

## Developers

### Code Checks

- Linting/formatting can use pre-commit
  ```commandline
  pre-commit run --all-files
  ```
