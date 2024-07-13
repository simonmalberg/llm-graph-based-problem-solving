import ast
import logging
from typing import Any, Dict, List, Union
from graph_of_thoughts import parser
from . import utils


class HotpotQAParser(parser.Parser):
    """
    HotpotQAParser provides the parsing of the language model responses specific to the gsm8k example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> Dict:
        new_states = []
        for text in texts:
            if state["method"].startswith("io"):
                new_state = state.copy()
                if "<Keywords>" in text:
                    text = text.split("<Keywords>")[1].split("</Keywords>")[0]
                    try:
                        keywords_list = ast.literal_eval(text)
                    except Exception as e:
                        logging.error(f"Could not parse keywords: {text}")
                        keywords_list = []
                    new_state["keywords"] = keywords_list
                new_state["current"] = text
                new_state["phase"] = state["phase"] + 1
                new_states.append(new_state)
            elif state["method"].startswith("probtree"):
                new_state = state.copy()
                new_state["current"] = text
                new_state["phase"] = state["phase"] + 1
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
        pass

    def parse_retrieve_answer(self, state: Dict, documents: Dict[Dict, Any]) -> List[Dict]:
        new_states = []
        documents_as_text = ""
        for key, value in documents.items():
            documents_as_text += f"Search Term: {key}\n<documents>\n"
            for doc in value[0][0]:
                documents_as_text += f"## {doc['title']}\n{"".join(doc['text'])}\n"
            documents_as_text += "</documents>\n"
        new_state = state.copy()

        new_state["current"] = documents_as_text
        new_state["phase"] = state["phase"] + 1
        new_states.append(new_state)
        return new_states