from typing import Literal

from pydantic import BaseModel, Field

from ts_bench.task_bank import TaskDefinition


class SubmissionFormat(BaseModel):
    file_extension: Literal[".npy"]
    shape: str = Field(
        ...,
        description="Expected tensor shape, e.g. [N, T, D]",
    )
    dtype: Literal["float32"]


class AssignmentMessage(BaseModel):
    instruction: str = Field(
        ...,
        description="Human-readable instructions for completing the task",
        min_length=50,
    )
    task_specification: TaskDefinition
    submission_format: SubmissionFormat


def create_assignment_message(
    assignment_num: int, assignment: TaskDefinition
) -> AssignmentMessage:
    n = assignment_num + 1

    instruction_text = (
        f"This is your assignment {n}. "
        "You will receive a Kaggle Dataset public URL to download the task bundle. "
        "The dataset contains the following files:\n"
        "- dataset.zip: training, validation, and test data\n"
        "- eval_fn.py: evaluation metrics and scoring code\n"
        "- task.txt: detailed task description and requirements\n\n"
        "You are expected to:\n"
        "1. Download and analyze the task bundle\n"
        "2. Develop, train, and tune your model using the training and validation data\n"
        "3. Generate predictions for the test data\n"
        "4. Save your predictions in the required format and return ONLY the file path\n\n"
        "Important notes:\n"
        "- The submission must be a single `.npy` file\n"
        "- The file must contain exactly one array representing the full prediction tensor\n"
        "- Do NOT return CSV, TXT, PKL, or any other file format"
    )

    message = AssignmentMessage(
        instruction=instruction_text,
        task_specification=assignment,
        submission_format=SubmissionFormat(
            file_extension=".npy",
            shape="[N, T, D]",
            dtype="float32",
        ),
    )
    return message
