import os
import re
import logging
import datetime
import json
from pathlib import Path
from typing import Dict, List, Callable, Union
from collections import Counter
from graph_of_thoughts import controller, language_models, operations, prompter, parser
from graph_of_thoughts.operations.thought import Thought

# This is a hack to also allow execution of this file from the examples directory
try:
    from . import utils
except ImportError:
    import utils


class GSM8KPrompter(prompter.Prompter):
    """
    GSM8KPrompter provides the generation of prompts specific to the gsm8k example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    io_prompt_base = """\
    <Instruction> Solve the following math problems and provide ONLY the integer solution with no comma or dot. Do not output any thoughts, only the answer after. </Instruction>

    <Examples>
    Input: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
    Output: 72

    Input: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
    Output: 10

    Input: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
    Output: 5

    Input: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
    Output: 42

    Input: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
    Output: 624
    </Examples>

    Input: {input}
    Output: """

    cot_prompt_base = """\
    <Instruction> Solve the following math problems and provide the full reasoning in the answer as well as the integer solution behind ####  with no comma or dot. </Instruction>

    <Examples>
    Input: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
    Output: Natalia sold 48/2 = 24 clips in May.
    Natalia sold 48+24 = 72 clips altogether in April and May.
    #### 72

    Input: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
    Output: Weng earns 12/60 = $0.2 per minute.
    Working 50 minutes, she earned 0.2 x 50 = $10.
    #### 10

    Input: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
    Output: In the beginning, Betty has only 100 / 2 = $50.
    Betty's grandparents gave her 15 * 2 = $30.
    This means, Betty needs 100 - 50 - 30 - 15 = $5 more.
    #### 5

    Input: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
    Output: Maila read 12 x 2 = 24 pages today.
    So she was able to read a total of 12 + 24 = 36 pages since yesterday.
    There are 120 - 36 = 84 pages left to be read.
    Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = 42 pages.
    #### 42

    Input: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
    Output: He writes each friend 3*2=6 pages a week.
    So he writes 6*2=12 pages every week.
    That means he writes 12*52=624 pages a year.
    #### 624
    </Examples>

    Input: {input}
    Output: """

    score_prompt_base = """\
    <Instruction> Score the answer given for the following (partial) math problem. The answer should be binary, True if the answer is correct and False if it is incorrect. Output only True or False. </Instruction>
    Input (partial) math problem: {input}
    Answer to the math problem: {answer}
    Output: """

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
        assert num_branches == 1, "Branching should be done via multiple requests."
        if current is None or current == "":
            input = original
        else:
            input = current
        
        if method.startswith("io"):
            return self.io_prompt_base.format(input=input)
        elif method.startswith("cot"):
            return self.cot_prompt_base.format(input=input)
        else:
            raise ValueError(f"Unknown method: {method}")
        

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        if len(state_dicts) > 1:
            raise NotImplementedError("Scoring multiple states is not implemented.")
        return self.score_prompt_base.format(input=state_dicts[0]["original"], answer=state_dicts[0]["current"])

    def improve_prompt(self, **kwargs) -> str:
        pass

    def validation_prompt(self, **kwargs) -> str:
        pass
    
    
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
            if state["method"].startswith("io") or state["method"].startswith("cot"):
                int_answer = utils.strip_int_result(text, state["method"])
                if int_answer is not None:
                    logging.warning(
                        f"Could not parse step answer: {text}. Returning None."
                    )
                new_state = state.copy()
                new_state["current"] = int_answer
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

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        assert len(states) == 1, "Scoring multiple states is not implemented."
        score = []
        for text in texts:
            if "True" in text and not "False" in text:
                score.append(1.0)
            else:
                score.append(0.0)
        return score


def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(1, False))
    operations_graph.append_operation(operations.GroundTruth(utils.test_answer))

    return operations_graph

def cot() -> operations.GraphOfOperations:
    return io()

def cotsc() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the CoT method with Self-Consistency.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 5))
    operations_graph.append_operation(operations.ScoreByFrequency(ignore_none=True))
    operations_graph.append_operation(operations.KeepBestN(1, True))
    operations_graph.append_operation(operations.Score(1, False))
    operations_graph.append_operation(operations.GroundTruth(utils.test_answer))

    return operations_graph

def run(
        data_ids: List[int],
        methods: List[Callable[[], operations.GraphOfOperations]],
        budget: float,
        lm_name: str,
    ) -> float:

    orig_budget = budget
    data_path = Path.cwd() / "datasets" / "grade-school-math" / "grade_school_math" / "data" / "test.jsonl"
    data = []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            json_line = json.loads(line)
            json_line["id"] = i
            data.append(json_line)

    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]

    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"{extra_info}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder)
    config = {
        "data": selected_data,
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    with open(os.path.join(results_folder, "config.json"), "w") as f:
        json.dump(config, f)
    
    logging.basicConfig(
        filename=os.path.join(results_folder, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    for method in methods:
        # create a results directory for the method
        os.makedirs(os.path.join(results_folder, method.__name__))
    
    for data in selected_data:
        logging.info(f"Running data {data['id']: }{data['question']}: {data['answer']}")
        if budget <= 0.0:
            logging.error(
                f"Budget has been depleted, stopping. Data {data[0]} has not been run."
            )
            break
        for method in methods:
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
                GSM8KPrompter(),
                GSM8KParser(),
                {
                    "original": data["question"],
                    "ground_truth": utils.strip_int_result(data["answer"]),
                    "current": "",
                    "phase": 0,
                    "method": method.__name__,
                },
            )
            try:
                executor.run()
            except Exception as e:
                logging.error(f"Exception: {e}")
            path = os.path.join(
                results_folder,
                method.__name__,
                f"{data['id']}.json",
            )
            executor.output_graph(path)
            budget -= lm.cost

    return orig_budget - budget


if __name__ == "__main__":
    budget = 30
    samples = [item for item in range(5)]
    approaches = [cotsc]

    logging.basicConfig(level=logging.INFO)

    spent = run(samples, approaches, budget, "chatgpt")

    logging.info(f"Spent {spent} out of {budget} budget.")
    