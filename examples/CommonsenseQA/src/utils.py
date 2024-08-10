import re
import re
import logging
from pathlib import Path
from typing import Dict, List, Callable, Union


def extract_answer(text: str):
    match = re.search(r'<Answer>(.*?)<\/Answer>', text)
    if match:
        return match.group(1)
    else:
        logging.error(f"extract_answer: unable to extract answer from {text}")
    return None


min_score = 1
max_score = 5
scoring_range = range(min_score, max_score + 1)


def extract_score(score_range: range, text: str):
    match = re.search(r'<Score>(.*?)<\/Score>', text)
    if match:
        try:
            score_val = int(match.group(1))
            if score_val in range(1, 6):
                return score_val
            else:
                logging.warning(f"Scored value is not in the range of {scoring_range} for response {text}")
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
        ground_truth = state["ground_truth"].strip().lower()
        current_answer = state["current"].strip().lower()
        return ground_truth == current_answer
    except:
        return False


def generate_question_string(question: dict):
    # Assemble a list of choice strings formatted with labels and text
    choices = [f'{choice["label"]} {choice["text"]}' for choice in question['choices']]
    # Combine the stem and choices into a single string
    question_text = question["stem"] + "\n" + "\n".join(choices)
    return question_text
