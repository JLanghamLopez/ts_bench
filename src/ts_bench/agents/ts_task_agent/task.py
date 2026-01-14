from pydantic import BaseModel, Field

from ts_bench.task_bank import TaskDefinition, TaskType


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
    instruction_text: str = (
        f"This is assignment number {assignment_num + 1}.\n\n"
        "You are provided with a task specification that defines the task objective, "
        "dataset access links, the evaluation function link, and the required output format.\n\n"
        "You are expected to follow the task description carefully and ensure that your submission "
        "strictly conforms to the specified requirements.\n\n"
    )

    if assignment.task_type == TaskType.TIME_SERIES_FORECASTING:
        instruction_text += (
            "This is a time-series forecasting task.\n\n"
            "You will be provided with:\n"
            "- Datasets accessible via the `data_urls` field\n"
            "- An evaluation function accessible via the `eval_url` field\n\n"
            "Your objectives are:\n"
            "1. Download the datasets using the provided `data_url` values\n"
            "2. Develop, train, and tune a forecasting model using the training data\n"
            "3. Generate predictions for the required inputs as defined in the task description\n"
            "4. Submit your predictions in the required JSON format\n\n"
            "Submission format (STRICT):\n"
            '- The submission must be a JSON object with exactly one key: "predictions".\n'
            '- The value of "predictions" must be a nested list of numerical values.\n'
            "- The shape of the predictions must exactly match the output_shape specified in the task.\n"
        )

    else:
        instruction_text += (
            "This is a time-series generation task.\n\n"
            "You will be provided with:\n"
            "- A training dataset accessible via the `data_urls` field\n"
            "- An evaluation function accessible via the `eval_url` field\n\n"
            "Your objectives are:\n"
            "1. Download the training data using the provided `data_urls`\n"
            "2. Develop, train, and tune a generative model using the training data\n"
            "3. Generate synthetic samples according to the task description\n"
            "4. Submit the generated samples in the required JSON format\n\n"
            "Submission format (STRICT):\n"
            '- The submission must be a JSON object with exactly one key: "predictions".\n'
            '- The value of "predictions" must be a nested list of numerical values.\n'
            "- The shape of the generated samples must exactly match the output_shape specified in the task.\n"
        )

    message = AssignmentMessage(
        instruction=instruction_text,
        task_specification=assignment,
    )
    return message
