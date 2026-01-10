from pydantic import BaseModel, Field

from ts_bench.task_bank import TaskDefinition


class AssignmentMessage(BaseModel):
    instruction: str = Field(
        ...,
        description="Human-readable instructions for completing the task",
        min_length=50,
    )
    task_specification: TaskDefinition


def create_assignment_message(
    assignment_num: int, assignment: TaskDefinition
) -> AssignmentMessage:
    instruction_text = (
        f"This is assignment number: {assignment_num + 1}."
        "You will receive a URL to download the task bundle. "
        "The task bundle contains the following files:\n"
        "- dataset.zip: training, validation, and test datasets\n"
        "- eval_fn.py: evaluation metrics and scoring code\n"
        "- task.txt: detailed task description and requirements\n\n"
        "You are expected to:\n"
        "1. Download and analyze the task bundle\n"
        "2. Develop, train, and tune your model using the training and validation data\n"
        "3. Generate predictions for the test data\n"
        "4. Submit your predictions in the required JSON format\n\n"
        "JSON requirements (STRICT):\n"
        '- The JSON object must have exactly one key: "predictions".\n'
        '- The value of "predictions" must be an array represented by nested Python lists.\n'
    )

    message = AssignmentMessage(
        instruction=instruction_text,
        task_specification=assignment,
    )
    return message
