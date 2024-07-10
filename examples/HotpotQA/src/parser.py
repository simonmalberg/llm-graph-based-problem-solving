import logging
from typing import Dict, List, Union
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
        pass
    
    def parse_aggregation_answer(self, response: str, **kwargs) -> Union[Dict, List[Dict]]:
        pass

    def parse_improve_answer(self, response: str, **kwargs) -> Dict:
        pass

    def parse_validation_answer(self, response: str, **kwargs) -> bool:
        pass

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        pass