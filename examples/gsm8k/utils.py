import re
from typing import Dict


def strip_int_result(text: str, method: str = "not_io") -> int:
        if method.startswith("io"):
            match = re.search(r'(\d+).*', text, re.DOTALL)
        else:
            match = re.search(r'#### (\d+).*', text, re.DOTALL)
        if match:
            return int(match.group(1))
        return None


def test_answer(state: Dict) -> bool:
    """
    Function to test whether the final solution matches ground truth.

    :param state: Thought state that represents the final solution.
    :type state: Dict
    :return: Returns whether the solution matches the ground truth.
    :rtype: bool
    """

    try:
        ground_truth = state["ground_truth"]
        current_answer = state["current"]
        return ground_truth == current_answer
    except:
        return False
    

def percentage_off(state: Dict) -> float:
    """
    Function to calculate how far off the answer is from the ground truth.

    :param state: Thought state to be scored.
    :type state: Dict
    :return: Number of errors.
    :rtype: float
    """
    
    try:
        ground_truth = state["ground_truth"]
        current_answer = state["current"]
        return abs(ground_truth - current_answer) / ground_truth
    except:
        return 1.0
        