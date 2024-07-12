import logging
from typing import Dict, List, Union
from graph_of_thoughts import parser
from . import utils
import re


def extract_score(score_range: range, text: str):
    match = re.search(r'<Score>(.*?)<\/Score>', text)
    if match:
        try:
            score_val = int(match.group(1))
            if score_val in score_range:
                return score_val
            else:
                logging.warning(f"Scored value is not in the range of {score_range} for response {text}")
        except ValueError:
            logging.warning(f"Unable to parse score from response {text}, returning 0 as score")
    return 0.0


class GSM8KParser(parser.Parser):
    """
    GSM8KParser provides the parsing of the language model responses specific to the gsm8k example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the response from the language model.
        :rtype: List[Dict]
        """
        new_states = []
        for text in texts:
            logging.info(f"parse_generate_answer: {text}")
            if state["method"].startswith("io") or state["method"].startswith("cot") or state["method"].startswith(
                    "plan_and_solve"):
                int_answer = utils.strip_int_result(text, state["method"])
                if int_answer is not None:
                    logging.warning(
                        f"Could not parse step answer: {text}. Returning None."
                    )
                new_state = state.copy()
                new_state["current"] = int_answer
                new_state["phase"] = 2
                new_states.append(new_state)
            elif state["method"] == "tot":
                logging.info("parse_generate_answer: phase = {}".format(state["phase"]))
                if state["phase"] < 2:
                    new_state = state.copy()
                    new_state["current"] = text
                    new_state["phase"] = 2
                    new_states.append(new_state)
                else:
                    new_state = state.copy()
                    new_state["phase"] = 3
                    new_state["current"] = utils.strip_int_result(text, state["method"])
                    new_states.append(new_state)
            else:
                raise ValueError(f"Unknown method: {state['method']}")
        return new_states

    def parse_aggregation_answer(self, response: str, **kwargs) -> Union[Dict, List[Dict]]:
        pass

    def parse_improve_answer(self, response: str, **kwargs) -> Dict:
        pass

    def parse_validation_answer(self, response: str, **kwargs) -> bool:
        pass

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        assert len(states) == 1, "Scoring multiple states is not implemented."
        for text in texts:
            logging.info(f"parse_score_answer: {text}")
        method = states[0]["method"]
        if method == "tot":
            scores: List[float] = [float(extract_score(10, text)) for text in texts]
            logging.info("parse_score_answer: scores: %s", scores)
            return scores
        else:
            score = []
            for text in texts:
                if "True" in text and "False" not in text:
                    score.append(1.0)
                else:
                    score.append(0.0)
            return score
