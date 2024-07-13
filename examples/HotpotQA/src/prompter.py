from typing import Dict, List
from graph_of_thoughts import prompter


class HotpotQAPrompter(prompter.Prompter):
    """
    HotpotQAPrompter provides the generation of prompts specific to the hotpotqa example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    io_prompt_get_keywords = """\
<Instruction> Give me a list of keywords for a wikipedia lookup to be able to answer this question. Give the keywords in the following format: give the score in the format <Keywords>["keyword1", "keyword2"]</Keywords>.</Instruction>
<Question>{input}</Question>
Output:
"""

    io_prompt_answer_question = """\
<Instruction> Answer the question using the provided context. Only output the final answer directly without any other text.</Instruction>
<Context>{context}</Context>
<Question>{question}</Question>
Output:
"""


    def generate_prompt(self, num_branches: int, original: str, current: str, method: str, **kwargs) -> str:
        assert num_branches == 1, "Branching should be done via multiple requests."
        if method.startswith("io"):
            if kwargs["phase"] == 0:
                prompt = self.io_prompt_get_keywords.format(input=original)
            elif kwargs["phase"] == 2:
                prompt = self.io_prompt_answer_question.format(context=current, question=original)
        return prompt

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def improve_prompt(self, **kwargs) -> str:
        pass

    def validation_prompt(self, **kwargs) -> str:
        pass
    