from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import os
import re
import traceback
from pathlib import Path
from typing import Dict, List, Callable, Union, Any

from tqdm import tqdm

import project_utils as project
try:
    from .bbh_tasks import BBHTask as BBH_Tasks
except ImportError:
    from bbh_tasks import BBHTask as BBH_Tasks
from graph_of_thoughts import controller, language_models, operations, prompter, parser
from graph_of_thoughts.operations import Thought


def extract_answer(text: str):
    match = re.search(r'<Answer>(.*?)<\/Answer>', text)
    if match:
        return match.group(1)
    return None


def extract_step(text: str):
    match = re.search(r'<Step>(.*?)<\/Step>', text)
    if match:
        return match.group(1)
    return None


min_score = 1
max_score = 5
scoring_range = range(min_score, max_score + 1)


def extract_score(score_range: range, text: str):
    match = re.search(r'<Score>(.*?)<\/Score>', text)
    if match:
        try:
            score_val = int(match.group(1))
            if score_val in range(1, 5):
                return score_val
            else:
                logging.warning(f"Scored value is not in the range of {scoring_range} for response {text}")
        except ValueError:
            logging.warning(f"Unable to parse score from response {text}, returning 0 as score")
    return 0.0


# def score_answers_by_frequency(thoughts: List[Thought]) -> List[Thought]:
#     ignore_empty_answers = False  # Setting to true helps with Llama3 as it often doesn't give answer in parseable form.
#     scores = {}
#     for thought in thoughts:
#         current_state = thought.state["current"]
#         scores[current_state] = scores.get(current_state, 0) + 1
#     logging.info("return_most_frequent_answer: scores: {}".format(scores))
#     frequent_answer = max(scores, key=scores.get)
#     for thought in thoughts:
#         thought.score = scores[thought.state["current"]]
#         if ignore_empty_answers:
#             if thought.state is None:
#                 thought.score = 0
#     return thoughts


def test_answer(state: Dict) -> bool:
    logging.debug(f"\nground truth: {state['ground_truth']}\n current_answer: {state['current']}")
    if state["current"] is None:
        return False
    ground_truth = state["ground_truth"].strip().lower()
    current_answer = state["current"].strip().lower()
    return ground_truth == current_answer


class BigBenchHardPrompter(prompter.Prompter):
    """
    Generate Prompts for the BigBench-Hard Dataset.
    """

    sys_prompt = """Provide the answer in the exact format as given"""

    io_zeroshot_prompt = """<Instruction> {instruction} </Instruction>
    
<Input> {input} </Input>"""

    io_prompt = """<Instruction> {instruction} </Instruction>
    
<Examples>
{examples}
</Examples>
    
<Input> {input} </Input>"""

    answer_prompt = """<Answer>{answer}</Answer>"""

    # Zero Shot COT from Kojima et al. (2022)
    cot_zeroshot_prompt = """<Instruction> {instruction} </Instruction>
    
    <Input> {input} </Input>
    Let us think Step by Step, then provide the answer in this format: <Answer>answer</Answer>"""

    tot_generate_prompt = """<Instruction> {instruction} </Instruction>
    <Input>{input}</Input>
    give a possible intermediate step towards the solution in the format: <Step>intermediate solution</Step>
    """

    tot_vote_step_prompt = f"""Given the question <Input>{{input}}</Input>
    Score the given intermediate step from {min_score} to {max_score} and give the score in the format <Score>score</Score> 
    <Step>{{step}}</Step>
    """

    tot_vote_final_prompt = f"""Given the question <Input>{{input}}</Input>
        Score the given answer from {min_score} to {max_score} and give the score in the format <Score>score</Score> 
        <Answer>{{answer}}</Answer>
        """

    tot_final_prompt = """Given the question <Input>{input}</Input> and the intermediate solution <Step>{step}</Step>, 
    provide the answer in this format: <Answer>answer</Answer>"""

    instruction_prefix = """<Instruction> {instruction} </Instruction> 
    "<Input> {input} </Input>
    """

    # The Plan and Solve Prompt from Wang et al. (2023)
    plan_solve_basic_prompt = "Let's first understand the problem and devise a plan to solve the problem. " \
                              "Then, let's carry out the plan to solve the problem step by step. " \
                              "Give the final answer in this format: <Answer>answer</Answer>"

    # The Plan and Solve Plus Prompt from Wang et al. (2023)
    plan_solve_plus_prompt = "Let's first understand the problem, extract relevant variables and their corresponding numerals, " \
                             "and make and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables " \
                             "(pay attention to correct numerical calculation and commonsense), " \
                             "solve the problem step by step, and show the answer. " \
                             "Give the final answer in this format: <Answer>answer</Answer>"

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
        logging.info(f"generate_prompt: method {method}")
        with open(prompt_path, "r") as f:
            for _ in range(2):  # to skip canary warning
                f.readline()
            prompt: str = f.readline().rstrip("\n")  # prompt is always on the 3rd line for each task
            examples: List[str] = f.read().lstrip("\n").split("\n\n")  # the rest of the file constitutes the example
            answers: List[str] = []
            for example in examples:
                answers.append(example.split("So the answer is ")[-1].removesuffix("."))  # extract answer from example
            full_examples: List[str] = []
            for i, example in enumerate(examples):
                if method.startswith("io") or method.startswith("plan_solve"):
                    full_examples.append(example.split("\nA: ")[0])  # remove steps for IO
                elif method.startswith("cot"):
                    full_examples.append(example)
                elif method.startswith("tot") or method.startswith("got"):
                    full_examples = []
                else:
                    raise ValueError(f"generate_prompt: Unknown method: {method}")

                full_examples.append(self.answer_prompt.format(answer=answers[i]))

            if method == "io":
                full_prompt = self.io_prompt.format(instruction=f"{self.sys_prompt}\n{prompt}",
                                                    examples="\n".join(full_examples),
                                                    input=input_str)
            elif method == "io_zs":
                full_prompt = self.io_zeroshot_prompt.format(instruction=f"{self.sys_prompt}\n{prompt}",
                                                    input=input_str)

            elif method == "cot" or method == "cot_sc":
                full_prompt = self.io_prompt.format(instruction=f"{self.sys_prompt}\n{prompt}",
                                                    examples="\n".join(full_examples),
                                                    input=input_str)
            elif method == "cot_zeroshot":
                full_prompt = self.cot_zeroshot_prompt.format(instruction=f"{self.sys_prompt}\n{prompt}",
                                                              input=input_str)
            elif method.startswith("tot") or method.startswith("got"):
                if current is None or current == "":
                    full_prompt = self.tot_generate_prompt.format(instruction=f"{self.sys_prompt}\n{prompt}",
                                                                  input=input_str)
                else:
                    full_prompt = self.tot_final_prompt.format(input=original, step=input_str)
            elif method == "plan_solve":
                full_prompt = self.instruction_prefix.format(instruction=f"{self.sys_prompt}\n{prompt}", input=input_str)+self.plan_solve_basic_prompt
            elif method == "plan_solve_plus":
                full_prompt = self.instruction_prefix.format(instruction=f"{self.sys_prompt}\n{prompt}", input=input_str)+self.plan_solve_plus_prompt
            else:
                raise ValueError(f"generate_prompt: Unknown method: {method}")

        logging.info("generate_prompt: full prompt: %s", full_prompt)
        return full_prompt

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def improve_prompt(self, **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        thoughts: List[Thought] = [state_dict["current"] for state_dict in state_dicts]
        inputs = [state_dict["original"] for state_dict in state_dicts]
        logging.info("score_prompt st_dicts: %s", json.dumps(state_dicts))
        final_prompt = ""
        if state_dicts[0]["phase"] < 2:
            final_prompt = self.tot_vote_step_prompt.format(step=thoughts[0], input=inputs[0])
        else:
            final_prompt = self.tot_vote_final_prompt.format(answer=thoughts[0], input=inputs[0])
        logging.info("score_prompt: final prompt: %s", final_prompt)
        return final_prompt

    def validation_prompt(self, **kwargs) -> str:
        pass


class BigBenchHardParser(parser.Parser):
    """
    BigBenchHardParser provides the parsing of the language model responses to the BigBench-Hard Dataset.

    Inherits from the Parser class and implements its abstract methods.
    """

    def parse_retrieve_answer(self, state: Dict, documents: Dict[Dict, Any]) -> List[Dict]:
        pass

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
            if (state["method"].startswith("io")
                    or state["method"].startswith("cot")
                    or state["method"].startswith("plan_solve")):
                answer_str = extract_answer(text)
                if answer_str is None:
                    logging.warning(
                        f"Could not parse step answer: {text}. Returning None."
                    )
                new_state = state.copy()
                new_state["current"] = answer_str
                new_state["phase"] = 2
                new_states.append(new_state)
            elif state["method"].startswith("tot") or state["method"].startswith("got"):
                if state["phase"] == 0:
                    logging.info("parse_generate_answer: tot phase 0: extracting step")
                    step_str = extract_step(text)
                    if step_str is None:
                        logging.warning(
                            f"parse_generate_answer: tot: Could not parse step: {text}. Returning None."
                        )
                    new_state = state.copy()
                    new_state["current"] = step_str
                    new_state["phase"] = 1
                    new_states.append(new_state)
                elif state["phase"] == 1:
                    logging.info("parse_generate_answer: tot: phase 1: extracting answer")
                    answer_str = extract_answer(text)
                    if answer_str is None:
                        logging.warning(
                            f"parse_generate_answer: tot: Could not parse final answer: {text}. Returning None."
                        )
                    new_state = state.copy()
                    new_state["current"] = answer_str
                    new_state["phase"] = 2
                    new_states.append(new_state)
                else:
                    raise ValueError(f"parse_generate_answer: tot generate_prompt: Unknown phase: {state['phase']}")

            else:
                raise ValueError(f"parse_generate_answer: Unknown method: {state['method']}")
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

def io_zs() -> operations.GraphOfOperations:
    return io()


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


def cot_zeroshot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the COT ZeroShot method.

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
        operations.ScoreByFrequency(ignore_none=True)
    )
    operations_graph.append_operation(operations.KeepBestN(1))
    operations_graph.append_operation(operations.GroundTruth(test_answer))

    return operations_graph


def plan_solve() -> operations.GraphOfOperations:
    """
         Generates the Graph of Operations for the Plan and Solve method.

         :return: Graph of Operations
         :rtype: GraphOfOperations
         """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.GroundTruth(test_answer))

    return operations_graph


def plan_solve_plus() -> operations.GraphOfOperations:
    """
         Generates the Graph of Operations for the Plan and Solve Plus method.

         :return: Graph of Operations
         :rtype: GraphOfOperations
         """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.GroundTruth(test_answer))

    return operations_graph


def tot() -> operations.GraphOfOperations:
    """
     Generates the Graph of Operations for the TOT method.

     :return: Graph of Operations
     :rtype: GraphOfOperations
     """
    num_branches = 5
    keep_best = 2

    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, num_branches).named("Generate Intermediate Step"))
    operations_graph.append_operation(operations.Score().named("Score Intermediate Step"))
    operations_graph.append_operation(operations.KeepBestN(keep_best).named("Keep Best Intermediate Steps"))
    operations_graph.append_operation(operations.Generate(1, 1).named("Generate Answers Based on Intermediate Step"))
    operations_graph.append_operation(operations.Score().named("Score Answers"))
    operations_graph.append_operation(operations.KeepBestN(1).named("Keep Best Answer"))
    operations_graph.append_operation(operations.GroundTruth(test_answer).named("Evaluate GroundTruth"))
    return operations_graph


def got() -> operations.GraphOfOperations:
    """
     Generates the Graph of Operations for the GOT method.

     :return: Graph of Operations
     :rtype: GraphOfOperations
     """
    num_branches = 5
    keep_best = 2

    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, num_branches).named("Generate Intermediate Step"))
    operations_graph.append_operation(operations.Score().named("Score Intermediate Step"))
    operations_graph.append_operation(operations.KeepBestN(keep_best).named("Keep Best Intermediate Steps"))
    operations_graph.append_operation(operations.Generate(1, num_branches).named("Generate Answers Based on Intermediate Step"))
    operations_graph.append_operation(operations.ScoreByFrequency(ignore_none=True).named("Score Answers by Frequency"))
    operations_graph.append_operation(operations.KeepBestN(1).named("Keep Most Frequent Answer"))
    operations_graph.append_operation(operations.GroundTruth(test_answer).named("Evaluate GroundTruth"))
    return operations_graph


def run(
        methods: List[Callable[[], operations.GraphOfOperations]],
        budget: float,
        lm_name: str,
        tasks: List[str] = [],
        samples_per_task: int = None,
        use_dir: str = None
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
    if not use_dir:
        results_dir = project.create_results_dir(
            directory=os.path.dirname(__file__),
            lm_name=lm_name,
            methods=methods,
            config=config,
            tasks=tasks
        )
    else:
        results_dir = use_dir

    datasets_dir: Path = project.datasets_dir() / "BIG-Bench-Hard" / "bbh"
    for task in tqdm(tasks, desc="Tasks"):
        logging.info(f"Evaluating task: {task}")
        task_data_path: Path = datasets_dir / f"{task}.json"
        task_results_dir = results_dir / task
        with open(task_data_path, "r") as f:
            task_data = json.load(f)[
                "examples"]  # we load the entire json at once as it seems the files are not too big.
            for id, example in tqdm(enumerate(task_data), desc="Examples", total=len(task_data)):
                if samples_per_task and id >= samples_per_task:  # end evaluation when samples limit is reached
                    break
                for method in tqdm(methods, desc="Methods"):
                    method_results_dir = task_results_dir / method.__name__
                    output_path: Path = method_results_dir / f"{id}.json"
                    if output_path.exists():
                        logging.info(f"Skipping example {id} for method {method.__name__} as it already exists.")
                        continue
                    logging.info(f"Running method {method.__name__}")
                    logging.info(f"Budget left: {budget}")
                    if budget <= 0.0:
                        logging.error(
                            f"Budget has been depleted, stopping. Method {method.__name__} has not been run."
                        )
                        break
                    if lm_name.startswith("chatgpt"):
                        lm = language_models.ChatGPT(
                            os.path.join(
                                os.path.dirname(__file__),
                                "../../graph_of_thoughts/language_models/config.json",
                            ),
                            model_name=lm_name,
                            cache=True,
                        )
                    elif lm_name.startswith("replicate"):
                        lm = language_models.ReplicateLanguageModel(
                            os.path.join(
                                os.path.dirname(__file__),
                                "../../graph_of_thoughts/language_models/config.json",
                            ),
                            model_name=lm_name,
                            cache=True,
                        )
                    else:
                        raise ValueError(f"Unknown LM: {lm_name}")
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
                        logging.error("Trace: {}".format(traceback.format_exc()))
                    executor.output_graph(str(output_path))
                    budget -= lm.cost

    return orig_budget - budget


if __name__ == "__main__":
    budget = 30
    samples = None # runs all samples
    approaches = [io, io_zs, cot, cot_zeroshot, cot_sc, tot, plan_solve, plan_solve_plus, got]
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S'
    )

    # tasks = [task.value for task in [
    #     BBH_Tasks.BOOLEAN_EXPRESSIONS,
    #     BBH_Tasks.DYCK_LANGUAGES,
    # ]]
    tasks = []

    spent = run(approaches, budget, "chatgpt", tasks, samples)

    logging.info(f"Spent {spent} out of {budget} budget.")
