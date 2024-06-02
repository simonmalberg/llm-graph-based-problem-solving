import os
import re
import logging
import datetime
import json
from pathlib import Path
from typing import Dict, List, Callable, Union, Any
from graph_of_thoughts import controller, language_models, operations, prompter, parser
import project_utils as Project

TASKS: list[str] = ["boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
                    "dyck_languages",
                    "formal_fallacies", "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
                    "logical_deduction_seven_objects", "logical_deduction_three_objects", "movie_recommendation",
                    "multistep_arithmetic_two", "navigate", "object_counting", "penguins_in_a_table",
                    "reasoning_about_colored_objects", "ruin_names", "salient_translation_error_detection", "snarks",
                    "sports_understanding", "temporal_sequences", "tracking_shuffled_objects_five_objects",
                    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects", "web_of_lies",
                    "word_sorting"]
"""
TASKS: list[str] contains the list of 27 tasks specified in the BigBench-Hard Dataset.
"""


class BigBenchHardPrompter(prompter.Prompter):
    """
    Generate Prompts for the BigBench-Hard Dataset.
    """

    io_prompt = """<Instruction> {instruction} </Instruction>
    
<Examples>
{examples}
</Examples>
    
<Input> {input} </Input>
    """

    answer_prompt = """<Answer>{answer}</Answer>"""

    def generate_prompt(self, num_branches: int, method: str, task: str, **kwargs) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param method: Method for which the generate prompt is generated.
        :type method: str
        :param task: The task to generate a prompt for.
        :type task: str
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        :raise AssertionError: If the requested number of branches is not one.
        """
        prompt_path = Project.datasets_dir() / "BIG-Bench-Hard" / "cot-prompts" / f"{task}.txt"
        full_prompt: str = ""
        with open(prompt_path, "r") as f:
            for _ in range(2):  # to skip canary warning
                f.readline()
            prompt: str = f.readline().rstrip("\n")  # prompt is always on the 3rd line for each task
            examples: [str] = f.read().lstrip("\n").split("\n\n")  # the rest of the file constitutes the example
            answers: [str] = []
            for example in examples:
                answers.append(example.split("So the answer is ")[-1].removesuffix("."))  # extract answer from example
            full_examples: [str] = []
            for i, example in enumerate(examples):
                if method.startswith("io"):
                    full_examples.append(example.split("\nA: ")[0])  # remove steps for IO
                elif method.startswith("cot"):
                    full_examples.append(example)
                else:
                    logging.error("Unknown method %s", method)
                    full_prompt = "ERROR: Unknown method {}".format(method) # TODO only here for testing, should be removed later
                    return full_prompt

                full_examples.append(self.answer_prompt.format(answer=answers[i]))
            if method.startswith("io"): # TODO consider making methods an enum?
                full_prompt = self.io_prompt.format(instruction=prompt, examples="\n".join(full_examples), input="input")
            elif method.startswith("cot"):
                full_prompt = self.io_prompt.format(instruction=prompt, examples="\n".join(full_examples), input="input")
            else:
                logging.error("Unknown method %s", method)
                full_prompt = "ERROR: Unknown method {}".format(method)
        return full_prompt

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def improve_prompt(self, **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def validation_prompt(self, **kwargs) -> str:
        pass


def run(
        data_ids: List[int],
        methods: List[Callable[[], operations.GraphOfOperations]],
        budget: float,
        lm_name: str,
        tasks: List[str],
) -> float:
    orig_budget = budget
    if not tasks:
        logging.info("No tasks specified, selecting all tasks")
        tasks.extend(TASKS)


if __name__ == "__main__":
    prompter = BigBenchHardPrompter()
    for task in TASKS:
        print(prompter.generate_prompt(1, method="io", task=task))
