from data.task_bank import TaskDefinition


def create_assignment_message(assignment_num: int, assignment: TaskDefinition) -> str:
    """
    Create a textual instruction message for the purple agent.
    This tells them about the tasks and what's expected.
    """
    n = assignment_num + 1
    task_type = assignment.task_type.replace("-", " ").title()
    msg_lines = [
        f"# Time Series Benchmark - {task_type} task number {n}",
        "",
        "## Instructions",
        "",
        "For the task, you will receive a URL to download the task bundle.",
        "The bundle contains:",
        "- Training data",
        "- Validation data",
        "- Test data",
        "- Evaluation metrics code",
        "- Task description and requirements",
        "",
        "You are expected to:",
        "1. Download and analyze each task bundle",
        "2. Develop, train, and tune your model on the training and validation data",
        "3. Generate predictions for the test data",
        "4. Write your predictions to a CSV file, and return a path to the file." "",
        "## Task Specification",
        "",
        f"### Task {n}: {assignment.name}",
        f"- **Task ID**: {assignment.task_id}",
        f"- **Type**: {assignment.task_type.value}",
        f"- **Difficulty**: {assignment.difficulty.value}",
        f"- **Data URL**: {assignment.url}",
        "",
        "## Submission Format",
        "",
        "Please return ONLY the path to a csv containing your submission",
        "",
    ]

    return "\n".join(msg_lines)
