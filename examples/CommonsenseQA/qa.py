import os
import re
import logging
import datetime
import json
from pathlib import Path
from typing import Dict, List, Callable, Union
from graph_of_thoughts import controller, language_models, operations, prompter, parser
import project_utils as project


def extract_answer(text: str):
    match = re.search(r'<Answer>(.*?)<\/Answer>', text)
    if match:
        return match.group(1)
    else:
        logging.error(f"extract_answer: unable to extract answer from {text}")
    return None


def test_answer(state: Dict) -> bool:
    """
    Function to test whether the final solution matches ground truth.

    :param state: Thought state that represents the final solution.
    :type state: Dict
    :return: Returns whether the solution matches the ground truth.
    :rtype: bool
    """

    try:
        ground_truth = state["ground_truth"].strip().lower()
        current_answer = state["current"].strip().lower()
        return ground_truth == current_answer
    except:
        return False


class CommonsenseQAPrompter(prompter.Prompter):
    """
    GSM8KPrompter provides the generation of prompts specific to the gsm8k example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    io_prompt = """<Instruction> Use your commonsense knowledge to answer the following question. Choose the correct answer from the options provided below. Output the answer in the following format: 
    <Answer>answer</Answer>
    </Instruction>

    <Examples>
    Input: What item is most useful for keeping your pants from falling down?
    A) shirt
    B) belt
    C) watch
    D) shoes
    E) hat
    Output: <Answer>B</Answer>
    </Examples>

    Input: {input}
    Output:"""

    cot_zeroshot_prompt = """<Instruction> Use your commonsense knowledge to answer the following question. Let us think Step by Step, then provide the answer in this format: 
    <Answer>answer</Answer>
    e.g. <Answer>E</Answer>
    </Instruction>

    Input: {input}
    Output:"""

    cot_prompt = """<Instruction> Use your commonsense knowledge to answer the following question. Let us think Step by Step, then provide the answer in this format: <Answer>answer</Answer>
    
    <Approach>
    To give the best answer follow these steps:
    1.clearly state the letter of the answer in the format given.
    2.You need to first paraphrase the problem, state the relevant premise according to the context, 
    3.Deduct facts one at a step.Give the reason why the answer is correct.
    </Approach>
    <Examples>
    Input: Where is a business restaurant likely to be located?
    A) town
    B) at hotel
    C) mall
    D) business sector
    E) yellow pages
    Output:
    Business restaurants are designed to cater to individuals who are involved in business activities, often looking for convenience and efficiency during business hours.
    Town (A): Could include various zones, not necessarily linked to business activities.
    At Hotel (B): Could serve business travelers but is not exclusively for business professionals and might cater more to tourists.
    Mall (C): Focuses more on retail and family-oriented services rather than business dealings.
    Business Sector (D): Directly caters to the business crowd, located within or near business hubs and offices.
    Yellow Pages (E): Not a physical location but a business directory.
    Therefore the answer is 
    <Answer>D</Answer>

    Input: When someone doesn't know how to skate well, they normally do what to stay up?
    A) spin
    B) romance
    C) hold hands
    D) fall down
    E) grab side railing
    Output:
    Individuals who are not proficient in skating often need support to maintain balance and prevent falls. This support can come in various forms, depending on what is available and what the individual feels most comfortable with.
    Spin (A): This is a complex move usually performed by experienced skaters, not typical for beginners.
    Romance (B): This choice is unrelated to physical support for staying upright while skating.
    Hold hands (C): This is a common method for beginners to support each other and maintain balance.
    Fall down (D): This is a result of losing balance, not a method to stay up.
    Grab side railing (E): This provides physical support and is a common choice for beginners to help themselves stay upright.
    Therefore the answer is
    <Answer>E</Answer>
    </Examples>

    Input: {input}
    Output: """

    # The Plan and Solve Prompt from Wang et al. (2023)
    plan_solve_basic_prompt = """<Instruction> Use your commonsense knowledge to answer the following question. 
    Let's first understand the problem and devise a plan to solve the problem. 
    Then, let's carry out the plan to solve the problem step by step.
    Give the final answer in this format: <Answer>answer</Answer>
    e.g. <Answer>E</Answer>
    </Instruction>
    Input: {input}
    Output:"""

    # The Plan and Solve Plus Prompt from Wang et al. (2023)
    plan_solve_plus_prompt = """<Instruction> Use your commonsense knowledge to answer the following question. 
    Let's first understand the problem, extract relevant variables and their corresponding numerals, and make and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), 
    solve the problem step by step, and show the answer.
    Give the final answer in this format: <Answer>answer</Answer>
    e.g. <Answer>E</Answer>
    </Instruction>  
    
    Input: {input}
    Output:"""

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

        full_prompt = ""
        if method.startswith("io"):
            full_prompt = self.io_prompt.format(input=input)
        elif method == "cot" or method == "cot_sc":
            full_prompt = self.cot_prompt.format(input=input)
        elif method == "cot_zeroshot":
            full_prompt = self.cot_zeroshot_prompt.format(input=input)
        elif method == "plan_solve":
            full_prompt = self.plan_solve_basic_prompt.format(input=input)
        elif method == "plan_solve_plus":
            full_prompt = self.plan_solve_plus_prompt.format(input=input)
        else:
            raise ValueError(f"Unknown method: {method}")
        logging.info("full_prompt: %s", full_prompt)
        return full_prompt

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def improve_prompt(self, **kwargs) -> str:
        pass

    def validation_prompt(self, **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass


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
            #print(text)

            if (state["method"].startswith("io")
                    or state["method"].startswith("cot")
                    or state["method"].startswith("plan_solve")):
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
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.GroundTruth(test_answer))

    return operations_graph


def cot_zeroshot() -> operations.GraphOfOperations:
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


def generate_question_string(question: dict):
    choices = [f"{choice["label"]}) {choice["text"]}" for choice in question["choices"]]
    return f""" {question["stem"]}
{"\n".join(choices)}
"""


def run(
        data_ids: List[int],
        methods: List[Callable[[], operations.GraphOfOperations]],
        budget: float,
        lm_name: str,
) -> float:
    orig_budget = budget
    datasets_dir: Path = project.datasets_dir() / "CommonSenseQA"
    data_path = datasets_dir / "train_rand_split.jsonl"
    logging.info(f"data_path = {data_path}")
    data = []
    if os.path.exists(data_path):  # test if exits
        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                try:
                    json_line = json.loads(line)
                    json_line["id"] = i
                    data.append(json_line)
                except json.JSONDecodeError as e:
                    logging.error(f"Error decoding JSON on line {i}: {line}")  # print error
    else:
        logging.error("File does not exist.")

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

        logging.info(f"Running data {data['id']: }{data['question']}: {data['answerKey']}")
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
            raw_path = os.path.join(
                os.path.dirname(__file__),
                "../../graph_of_thoughts/language_models/config.json")
            abs_path = os.path.abspath(raw_path)
            lm = language_models.ChatGPT(
                abs_path,
                model_name=lm_name,
                cache=True,
            )

            operations_graph = method()
            executor = controller.Controller(
                lm,
                operations_graph,
                CommonsenseQAPrompter(),
                CommonsenseQAParser(),
                {
                    "original": generate_question_string(data["question"]),
                    "ground_truth": data["answerKey"],
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
    samples = [item for item in range(10)]
    approaches = [io, cot_zeroshot]
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S'
    )
    spent = run(samples, approaches, budget, "llama3-8b-ollama")

    logging.info(f"Spent {spent} out of {budget} budget.")
