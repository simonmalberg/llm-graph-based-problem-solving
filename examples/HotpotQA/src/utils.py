import ast
import logging
import re
import string
from pathlib import Path
from typing import List, Counter, Dict, Any

from graph_of_thoughts.operations import Thought


def extract_keywords(text: str):
    match = re.search(r'<Keywords>(.*?)<\/Keywords>', text)
    if match:
        try:
            keywords_list = ast.literal_eval(match.group(1))
            logging.info("Extracted keywords = {}".format(match.group(1)))
            return keywords_list
        except Exception as e:
            logging.error(f"Could not parse keywords: {text}")
    else:
        logging.error(f"Could not find <Keywords> tag: {text}")
    return None


def extract_answer(text: str):
    match = re.search(r'<Answer>(.*?)<\/Answer>', text)
    if match:
        logging.info("Extracted answer = {}".format(match.group(1)))
        return match.group(1)
    else:
        logging.error(f"Could not find <Answer> tag: {text}")
    return None


def keep_most_frequent_keywords(thoughts: List[Thought], keep_best: int) -> List[Thought]:
    states = [thought.state for thought in thoughts]
    keyword_lists = [state["keywords"] for state in states]
    aggregated_keywords: List[str] = []
    for list in keyword_lists:
        aggregated_keywords.extend(list)
    # aggregated_keywords = [keyword for klist in keyword_lists for keyword in klist]
    # logging.debug("keep_most_frequent_keywords: keywords_lists: {}".format("\n".join([", ".join(list) for list in keyword_lists])))
    counter = Counter[str](aggregated_keywords)
    most_common_keywords = [keyword for keyword, count in counter.most_common(keep_best)]
    logging.info(f"keep_most_frequent_keywords: most common {keep_best} keywords: {most_common_keywords}")
    thought = thoughts[0]
    thought.state["keywords"] = most_common_keywords
    return [thought]


def parse_examples() -> list[dict[str, str]]:
    question_answers = []
    with open(Path(__file__).resolve().parent / "probtree_prompt.txt", "r") as f:
        text = f.readlines()[1:]
        for i in range(0, len(text), 2):
            match = re.search(r"So the answer is: (.*)\.", text[i + 1])
            if match:
                answer = match.group(1)
            else:
                answer = ""
            item = {
                "question": text[i].removeprefix("Q: "),
                "cot_answer": text[i + 1].removeprefix("A: "),
                "io_answer": answer,
            }
            question_answers.append(item)
    return question_answers


def normalize_answer(s):
    """
    Code taken from the ProbTree repository (https://github.com/THU-KEG/ProbTree)
    @param s:
    @return:
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth) -> (float, float, float):
    """
        Code adapted from the ProbTree repository (https://github.com/THU-KEG/ProbTree)
        @param ground_truth:
        @param prediction:
        @return:
        """
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    zero_metric = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return zero_metric
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return zero_metric

    prediction_tokens: list[str] = normalized_prediction.split()
    ground_truth_tokens: list[str] = normalized_ground_truth.split()
    common = Counter[str](prediction_tokens) & Counter[str](ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return zero_metric
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def exact_match_score(prediction, ground_truth) -> bool:
    """
        Code taken from the ProbTree repository (https://github.com/THU-KEG/ProbTree)
        @param ground_truth:
        @param prediction:
        @return:
        """
    return normalize_answer(prediction) == normalize_answer(ground_truth)