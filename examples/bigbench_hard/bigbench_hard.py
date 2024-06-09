import os
import re
import logging
import datetime
import json
from pathlib import Path
from typing import Dict, List, Callable, Union, Any

from graph_of_thoughts.operations import Thought
from graph_of_thoughts import controller, language_models, operations, prompter, parser
import project_utils as project
from bbh_tasks import Tasks as BBH_Tasks


def extract_answer(text: str):
    match = re.search(r'<Answer>(.*?)<\/Answer>', text)
    if match:
        return match.group(1)
    return None


def score_answers_by_frequency(thoughts: List[Thought]) -> List[Thought]:
    ignore_empty_answers = False  # Setting to true helps with Llama3 as it often doesn't give answer in parseable form.
    scores = {}
    for thought in thoughts:
        current_state = thought.state["current"]
        scores[current_state] = scores.get(current_state, 0) + 1
    logging.info("return_most_frequent_answer: scores: {}".format(scores))
    frequent_answer = max(scores, key=scores.get)
    for thought in thoughts:
        thought.score = scores[thought.state["current"]]
        if ignore_empty_answers:
            if thought.state is None:
                thought.score = 0
    return thoughts


def test_answer(state: Dict) -> bool:
    logging.debug(f"\nground truth: {state['ground_truth']}\n current_answer: {state['current']}")
    ground_truth = state["ground_truth"]
    current_answer = state["current"]
    return ground_truth == current_answer


class BigBenchHardPrompter(prompter.Prompter):
    """
    Generate Prompts for the BigBench-Hard Dataset.
    """

    sys_prompt = """Provide the answer in the exact format as given"""
    io_prompt = """<Instruction> {instruction} </Instruction>
    
<Examples>
{examples}
</Examples>
    
<Input> {input} </Input>
    """

    answer_prompt = """<Answer>{answer}</Answer>"""

    def __init__(self, task: str):
        """
        @param task: The bigbench task to create prompts for.
        """
        self.task = task

    def generate_prompt(self, num_branches: int, original: str, current: str, method: str, **kwargs) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param original: Input text.
        :type original: str
        :param current: Intermediate solution.
        :type current: str
        :param method: Method for which the generate prompt is generated.
        :type method: str
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        :raise AssertionError: If the requested number of branches is not one.
        """

        if current is None or current == "":
            input_str = original
        else:
            input_str = current

        prompt_path = project.datasets_dir() / "BIG-Bench-Hard" / "cot-prompts" / f"{self.task}.txt"
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
                    raise ValueError(f"generate_prompt: Unknown method: {method}")

                full_examples.append(self.answer_prompt.format(answer=answers[i]))
            if method.startswith("io"):
                full_prompt = self.io_prompt.format(instruction=f"{self.sys_prompt}\n{prompt}",
                                                    examples="\n".join(full_examples),
                                                    input=input_str)
            elif method.startswith("cot"):
                full_prompt = self.io_prompt.format(instruction=f"{self.sys_prompt}\n{prompt}",
                                                    examples="\n".join(full_examples),
                                                    input=input_str)
            else:
                raise ValueError(f"generate_prompt: Unknown method: {method}")

        logging.info("full prompt: %s", full_prompt)
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
        new_states = []
        for text in texts:
            if state["method"].startswith("io") or state["method"].startswith("cot"):
                answer_str = extract_answer(text)
                if answer_str is None:
                    logging.warning(
                        f"Could not parse step answer: {text}. Returning None."
                    )
                new_state = state.copy()
                new_state["current"] = answer_str
                new_state["phase"] = 2
                new_states.append(new_state)
            else:
                raise ValueError(f"parse_generate_answer: Unknown method: {state['method']}")
        return new_states

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
    operations_graph.append_operation(operations.GroundTruth(test_answer))

    return operations_graph


def cot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the COT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.GroundTruth(test_answer))

    return operations_graph


def cot_sc() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the COT-SC method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    num_branches = 5
    operations_graph = operations.GraphOfOperations()

    generate_operation = operations.Generate(1, num_branches)
    operations_graph.append_operation(generate_operation)
    operations_graph.append_operation(
        operations.Selector(selector=score_answers_by_frequency))  # have to do this less than ideal implementation
    # due to issues with existing scoring function, might be better to refactor
    operations_graph.append_operation(operations.KeepBestN(1))
    operations_graph.append_operation(operations.GroundTruth(test_answer))

    return operations_graph


def run(
        data_ids: List[int],
        methods: List[Callable[[], operations.GraphOfOperations]],
        budget: float,
        lm_name: str,
        tasks: [str] = [],
) -> float:
    orig_budget = budget
    if not tasks:
        logging.info("No tasks specified, selecting all tasks")
        tasks.extend([task.value for task in list(BBH_Tasks)])
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
            task_data = json.load(f)[
                "examples"]  # we load the entire json at once as it seems the files are not too big.
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
                        BigBenchHardPrompter(task),
                        BigBenchHardParser(),
                        {
                            "original": example["input"],
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
                    output_path: Path = method_results_dir / f"{id}.json"
                    executor.output_graph(str(output_path))
                    budget -= lm.cost

    return orig_budget - budget


if __name__ == "__main__":
    budget = 30
    samples = [item for item in range(5)]
    approaches = [cot_sc]
    tasks = [task.value for task in [
        BBH_Tasks.BOOLEAN_EXPRESSIONS
    ]]

    spent = run(samples, approaches, budget, "llama3-8b-ollama")

    logging.info(f"Spent {spent} out of {budget} budget.")
