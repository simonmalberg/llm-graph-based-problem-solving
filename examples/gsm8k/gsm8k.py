import os
import logging
import datetime
import json
from pathlib import Path
from typing import List, Callable
from graph_of_thoughts import controller, language_models, operations
import project_utils as project

try:
    from .src.prompter import GSM8KPrompter
    from .src.parser import GSM8KParser
    from .src import utils
except ImportError:
    from src.prompter import GSM8KPrompter
    from src.parser import GSM8KParser
    from src import utils


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

def plan_and_solve_basic() -> operations.GraphOfOperations:
    return io()

def plan_and_solve_plus() -> operations.GraphOfOperations:
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


def tot() -> operations.GraphOfOperations:
    """
        Generates the Graph of Operations for the ToT method.

        :return: Graph of Operations
        :rtype: GraphOfOperations
        """
    num_branches = 5
    operations_graph = operations.GraphOfOperations()
    # Phase 1: setting up the equations
    operations_graph.append_operation(operations.Generate(1, num_branches).named("Generate Setup"))
    operations_graph.append_operation(operations.Score().named("Score Setup"))
    operations_graph.append_operation(operations.KeepBestN(1))
    # Phase 2: calculating the final results
    operations_graph.append_operation(operations.Generate(1, num_branches).named("Generate Final Answer"))
    operations_graph.append_operation(operations.Score().named("Score Final Answer"))
    operations_graph.append_operation(operations.KeepBestN(1))

    operations_graph.append_operation(operations.GroundTruth(utils.test_answer))
    return operations_graph


def got() -> operations.GraphOfOperations:
    """
        Generates the Graph of Operations for the GoT method.

        :return: Graph of Operations
        :rtype: GraphOfOperations
        """
    num_branches = 5
    operations_graph = operations.GraphOfOperations()
    # Phase 1: setting up the equations
    operations_graph.append_operation(operations.Generate(1, num_branches).named("Generate Setup"))
    operations_graph.append_operation(operations.Score().named("Score Setup"))
    operations_graph.append_operation(operations.KeepBestN(1))
    # Phase 2: calculating the final results
    operations_graph.append_operation(operations.Generate(1, num_branches).named("Generate Final Answer"))
    operations_graph.append_operation(operations.ScoreByFrequency(ignore_none=True).named("Score Final Answer By Frequency"))
    operations_graph.append_operation(operations.KeepBestN(1))

    operations_graph.append_operation(operations.GroundTruth(utils.test_answer))
    return operations_graph

def run(
        data_ids: List[int],
        methods: List[Callable[[], operations.GraphOfOperations]],
        budget: float,
        lm_name: str,
    ) -> float:

    orig_budget = budget
    data_path = project.datasets_dir() / "grade-school-math" / "grade_school_math" / "data" / "test.jsonl"
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
                    "ground_truth": utils.strip_int_result(data["answer"], is_groundtruth=True),
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
            # canvas_path = os.path.join(
            #     results_folder,
            #     method.__name__,
            #     f"{data['id']}.canvas",
            # )
            executor.output_graph(path)
            # executor.generate_json_canvas(canvas_path)
            budget -= lm.cost

    return orig_budget - budget


if __name__ == "__main__":
    budget = 30
    samples = [item for item in range(5)]
    approaches = [io, cot, cotsc, plan_and_solve_basic, plan_and_solve_plus]
    # approaches = [tot]

    logging.basicConfig(level=logging.INFO)

    spent = run(samples, approaches, budget, "chatgpt")

    logging.info(f"Spent {spent} out of {budget} budget.")
    