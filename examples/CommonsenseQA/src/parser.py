import logging
from typing import Dict, List, Union
from graph_of_thoughts import parser
from . import utils



class CommonsenseQAParser(parser.Parser):
    """
    CommonsenseQAParser provides the parsing of the language model responses specific to the CommonsenseQA example.

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
            
            if (state["method"].startswith("io")
                    or state["method"].startswith("cot")
                    or state["method"].startswith("plan_solve")
                    or state["method"].startswith("tot")):
                answer_key = extract_answer(text)
                if answer_key is None:
                    logging.warning(
                        f"Could not parse step answer: {text}. Returning None."
                    )
                new_state = state.copy()
                new_state["current"] = answer_key
                new_state["phase"] = 2
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

    def parse_score_answer(self, states: List[Dict], responses: List[str], **kwargs) -> List[float]:
        logging.info("parse_score_answer: responses: %s", responses)
        scores: List[float] = [float(extract_score(scoring_range, response)) for response in responses]
        logging.info("parse_score_answer: scores: %s", scores)
        return scores