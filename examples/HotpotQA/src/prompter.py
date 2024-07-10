from typing import Dict, List
from graph_of_thoughts import prompter


class HotpotQAPrompter(prompter.Prompter):
    """
    HotpotQAPrompter provides the generation of prompts specific to the hotpotqa example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """


    def generate_prompt(self, num_branches: int, original: str, current: str, method: str, **kwargs) -> str:
        pass

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def improve_prompt(self, **kwargs) -> str:
        pass

    def validation_prompt(self, **kwargs) -> str:
        pass
    