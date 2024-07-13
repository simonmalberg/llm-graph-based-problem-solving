import logging
import re
from typing import Dict


def strip_int_result(text: str, method: str = "not_io", is_groundtruth: bool = False) -> int:
        if is_groundtruth:
            match = re.search(r'#### (\d+).*', text, re.DOTALL)
        elif method.startswith("io"):
            match = re.search(r'(\d+).*', text, re.DOTALL)
        else:
            match = re.search(r'<Answer>(\d+)</Answer>', text, re.DOTALL)
        if match:
            return int(match.group(1))
        return None


def extract_int_answer(text: str):
    match = re.search(r'<Answer>(.*?)<\/Answer>', text)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            logging.error("Answer not parseable as int: {}".format(match.group(1)))
    else:
        logging.error("could not find answer tags in text {}".format(text))
    return None


def extract_score(score_range: range, text: str):
    match = re.search(r'<Score>(.*?)<\/Score>', text)
    if match:
        try:
            score_val = int(match.group(1))
            if score_val in range(10):
                return score_val
            else:
                logging.warning(f"Scored value is not in the range of {score_range} for response {text}")
        except ValueError:
            logging.warning(f"Unable to parse score from response {text}, returning 0 as score")
    return 0.0


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
        