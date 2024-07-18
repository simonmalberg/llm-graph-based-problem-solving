import ast
import logging
import re
from typing import List, Counter

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