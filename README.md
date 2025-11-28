# TS-Bench

## Developers

### Installation

Dependencies can be installed with [poetry](https://python-poetry.org/) by running

```commandline
poetry install
```

which will create a virtual environment in the repo at `.venv` which can
be activated with

```commandline
source .venv/bin/activate
```

### Code Checks

- Linting/formatting can be using pre-commit
  ```commandline
  pre-commit run --all-files
  ```
