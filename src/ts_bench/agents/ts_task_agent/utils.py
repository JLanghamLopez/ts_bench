import os


def check_response(response: str) -> None:
    """Check response is a path to a csv file"""
    assert os.path.exists(response)
    assert (
        response.split(".")[-1] == "csv"
    ), f"Predictions should be saved as a csv file, got {response}"
