import os
import re
import logging
import datetime
import json
from pathlib import Path
from typing import Dict, List, Callable, Union, Any
from graph_of_thoughts import controller, language_models, operations, prompter, parser
import project_utils as project

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
        prompt_path = project.datasets_dir() / "BIG-Bench-Hard" / "cot-prompts" / f"{task}.txt"
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
                    raise ValueError(f"Unknown method: {method}")

                full_examples.append(self.answer_prompt.format(answer=answers[i]))
            if method.startswith("io"):
                full_prompt = self.io_prompt.format(instruction=prompt, examples="\n".join(full_examples),
                                                    input="input")
            elif method.startswith("cot"):
                full_prompt = self.io_prompt.format(instruction=prompt, examples="\n".join(full_examples),
                                                    input="input")
            else:
                raise ValueError(f"Unknown method: {method}")

        return full_prompt

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def improve_prompt(self, **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def validation_prompt(self, **kwargs) -> str:
        pass


class BigBenchHardParser(parser.Parser):
    """
    BigBenchHardParser provides the parsing of the language model responses to the BigBench-Hard Dataset.

    Inherits from the Parser class and implements its abstract methods.
    """

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the response from the language model.
        :rtype: List[Dict]
        """
        raise NotImplementedError("This method needs to be implemented.")

    def parse_aggregation_answer(self, response: str, **kwargs) -> Union[Dict, List[Dict]]:
        pass

    def parse_improve_answer(self, response: str, **kwargs) -> Dict:
        pass

    def parse_validation_answer(self, response: str, **kwargs) -> bool:
        pass

    def parse_score_answer(self, response: str, **kwargs) -> List[float]:
        pass


def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.GroundTruth(NotImplemented))

    return operations_graph


def cot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the COT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.GroundTruth(NotImplemented))

    return operations_graph


def cot_sc() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the COT-SC method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    raise NotImplementedError("This method needs to be implemented.")


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
    else:
        logging.info("Running the following tasks: %s", tasks)

    # create the results directory
    config = {
        "tasks": tasks,
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }

    # create directory for results
    results_dir = project.create_results_dir(
        directory=os.path.dirname(__file__),
        lm_name=lm_name,
        methods=methods,
        config=config,
        tasks=tasks
    )

    datasets_dir: Path = project.datasets_dir() / "BIG-Bench-Hard" / "bbh"
    for task in tasks:
        logging.info(f"Evaluating task: {task}")
        task_data_path: Path = datasets_dir / f"{task}.json"
        task_results_dir = results_dir / task
        with open(task_data_path, "r") as f:
            task_data = json.load(f)["examples"]  # we load the entire json at once as it seems the files are not too big.
            for id, example in enumerate(task_data):
                for method in methods:
                    method_results_dir = task_results_dir / method.__name__
                    logging.info(f"Running method {method.__name__}")
                    logging.info(f"Budget left: {budget}")
                    if budget <= 0.0:
                        logging.error(
                            f"Budget has been depleted, stopping. Method {method.__name__} has not been run."
                        )
                        break
                    lm = language_models.ChatGPT(
                        os.path.join(
                            os.path.dirname(__file__),
                            "../../graph_of_thoughts/language_models/config.json",
                        ),
                        model_name=lm_name,
                        cache=True,
                    )
                    operations_graph = method()
                    executor = controller.Controller(
                        lm,
                        operations_graph,
                        BigBenchHardPrompter(),
                        BigBenchHardParser(),
                        {
                            "input": example["input"],
                            "ground_truth": example["target"],
                            "current": "",
                            "phase": 0,
                            "method": method.__name__
                        }
                    )
                    try:
                        executor.run()
                    except Exception as e:
                        logging.error(f"Exception: {e}")
                    output_path:Path = method_results_dir / f"{id}.json"
                    executor.output_graph(str(output_path))
                    budget -= lm.cost

    return orig_budget - budget

if __name__ == "__main__":
    budget = 30
    samples = [item for item in range(5)]
    approaches = [io]

    spent = run(samples, approaches, budget, "llama3-8b-ollama")

    logging.info(f"Spent {spent} out of {budget} budget.")
    # prompter = BigBenchHardPrompter()
    # for task in TASKS:
    #     print(prompter.generate_prompt(1, method="io", task=task))
