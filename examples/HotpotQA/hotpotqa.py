import datetime
import json
import logging
import os
from functools import partial
from typing import List, Callable, Dict

from graph_of_thoughts import controller, language_models, operations
from graph_of_thoughts.operations.probtree_operation import ProbtreeReasoning
from project_utils import datasets_dir

try:
    from .src.prompter import HotpotQAPrompter
    from .src.parser import HotpotQAParser
    from .src import utils
except ImportError:
    from src.prompter import HotpotQAPrompter
    from src.parser import HotpotQAParser
    from src import utils


def test_answer(state: Dict) -> bool:
    logging.debug(f"\nground truth: {state['ground_truth']}\n current_answer: {state['current']}")
    ground_truth = state["ground_truth"]
    current_answer = state["current"]
    return utils.exact_match_score(current_answer, ground_truth)


def calc_f1_score(state: Dict) -> float:
    score = utils.f1_score(state["current"], state["ground_truth"])
    return score[0]


def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(
        operations.Retrieve(bm25_retriever_save_dir=(datasets_dir() / "HotpotQA" / "wikipedia_index_bm25"), k=5))
    operations_graph.append_operation(operations.Generate(1, 1))
    # another generate process including the keywords and another prompt
    # groundtruth evaluation
    operations_graph.append_operation(operations.Score(scoring_function=calc_f1_score).named("Calculate F1 Score"))
    operations_graph.append_operation(operations.GroundTruth(test_answer))

    return operations_graph

def io_base() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO base method (retrieval based on the original question).

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(
        operations.Retrieve(
            bm25_retriever_save_dir=(datasets_dir() / "HotpotQA" / "wikipedia_index_bm25"),
            get_keywords_function=lambda state: [state["original"]],
            k=5))
    operations_graph.append_operation(operations.Generate(1, 1))
    # another generate process including the keywords and another prompt
    # groundtruth evaluation
    operations_graph.append_operation(operations.Score(scoring_function=calc_f1_score).named("Calculate F1 Score"))
    operations_graph.append_operation(operations.GroundTruth(test_answer))

    return operations_graph


def probtree() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ProbTree method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1, True))
    operations_graph.append_operation(ProbtreeReasoning(bm25_retriever_save_dir=(datasets_dir() / "HotpotQA" / "wikipedia_index_bm25")).named("ProbTree Reasoning"))
    # operations_graph.append_operation(operations.Retrieve(bm25_retriever_save_dir=(datasets_dir() / "HotpotQA" / "wikipedia_index_bm25"), k=5))
    # operations_graph.append_operation(operations.Generate(1, 1))
    # # another generate process including the keywords and another prompt
    # groundtruth evaluation
    operations_graph.append_operation(operations.Score(scoring_function=calc_f1_score).named("Calculate F1 Score"))
    operations_graph.append_operation(operations.GroundTruth(test_answer))

    return operations_graph


def cot() -> operations.GraphOfOperations:
    """
            Generates the Graph of Operations for the CoT Method.

            :return: Graph of Operations
            :rtype: GraphOfOperations
            """
    retrieval_count = 5
    operations_graph = operations.GraphOfOperations()
    operations_graph.append_operation(operations.Generate(1, 1).named("Generate Keywords"))
    operations_graph.append_operation(
        operations.Retrieve(bm25_retriever_save_dir=(datasets_dir() / "HotpotQA" / "wikipedia_index_bm25"),
                            k=retrieval_count).named("Retrieve Keywords"))
    operations_graph.append_operation(operations.Generate(1, 1).named("Generate Answer"))
    operations_graph.append_operation(operations.Score(scoring_function=calc_f1_score).named("Calculate F1 Score"))
    operations_graph.append_operation(operations.GroundTruth(test_answer))
    return operations_graph


def cot_sc_1() -> operations.GraphOfOperations:
    """
        Generates the Graph of Operations for the CoT-SC I Method.
        In this implementation, the retrieval and reasoning is done together `n` times, and self-consistency is implemented at the
        end for the `n` chains

        :return: Graph of Operations
        :rtype: GraphOfOperations
        """

    num_branches = 5
    retrieval_count = 5
    operations_graph = operations.GraphOfOperations()
    operations_graph.append_operation(operations.Generate(1, num_branches).named("Generate Keywords"))
    operations_graph.append_operation(
        operations.Retrieve(bm25_retriever_save_dir=(datasets_dir() / "HotpotQA" / "wikipedia_index_bm25"),
                            k=retrieval_count).named("Retrieve Keywords"))
    operations_graph.append_operation(operations.Generate(1, 1).named("Generate Answer"))
    operations_graph.append_operation(operations.ScoreByFrequency(ignore_none=True))
    operations_graph.append_operation(operations.KeepBestN(1))

    operations_graph.append_operation(operations.Score(scoring_function=calc_f1_score).named("Calculate F1 Score"))
    operations_graph.append_operation(operations.GroundTruth(test_answer))
    return operations_graph


def cot_sc_2() -> operations.GraphOfOperations:
    """
            Generates the Graph of Operations for the CoT-SC II Method.
            In this implementation, the retrieval is done `n` times, then self-consistency is implemented on a per key-word basis.
            Then, the frequent answer is used to retrieve documents, and the model is prompted `n` times for the final answer.
            Then, Self-consistency is applied to the final answer.

            :return: Graph of Operations
            :rtype: GraphOfOperations
            """
    num_branches = 5
    retrieval_count = 5
    operations_graph = operations.GraphOfOperations()
    operations_graph.append_operation(operations.Generate(1, num_branches).named("Generate Keywords"))
    operations_graph.append_operation(
        operations.Selector(selector=partial(utils.keep_most_frequent_keywords, keep_best=3)))
    operations_graph.append_operation(
        operations.Retrieve(bm25_retriever_save_dir=(datasets_dir() / "HotpotQA" / "wikipedia_index_bm25"),
                            k=retrieval_count).named("Retrieve Keywords"))
    operations_graph.append_operation(operations.Generate(1, num_branches).named("Generate Answer"))
    operations_graph.append_operation(operations.ScoreByFrequency(ignore_none=True))
    operations_graph.append_operation(operations.KeepBestN(1))
    operations_graph.append_operation(operations.Score(scoring_function=calc_f1_score).named("Calculate F1 Score"))
    operations_graph.append_operation(operations.GroundTruth(test_answer))
    return


def plan_solve_basic() -> operations.GraphOfOperations:
    return cot()


def plan_solve_plus() -> operations.GraphOfOperations:
    return cot()


def cot_zeroshot() -> operations.GraphOfOperations:
    return cot()


def tot() -> operations.GraphOfOperations:
    num_branches = 1
    retrieval_count = 5

    operations_graph = operations.GraphOfOperations()
    operations_graph.append_operation(operations.Generate(1, num_branches).named("Generate Keywords"))
    operations_graph.append_operation(
        operations.Retrieve(bm25_retriever_save_dir=(datasets_dir() / "HotpotQA" / "wikipedia_index_bm25"),
                            k=retrieval_count).named("Retrieve Keywords"))
    operations_graph.append_operation(operations.Generate(1, num_branches).named("Check Keywords or Answer"))
    operations_graph.append_operation(
        operations.Retrieve(bm25_retriever_save_dir=(datasets_dir() / "HotpotQA" / "wikipedia_index_bm25"),
                            k=retrieval_count).named("Retrieve Keywords"))
    operations_graph.append_operation(operations.Generate(1, num_branches).named("Final Answer"))
    operations_graph.append_operation(operations.Score(scoring_function=calc_f1_score).named("Calculate F1 Score"))
    operations_graph.append_operation(operations.GroundTruth(test_answer))
    return operations_graph


def run(
        data_ids: List[int],
        methods: List[Callable[[], operations.GraphOfOperations]],
        budget: float,
        lm_name: str,
) -> float:
    orig_budget = budget
    data_path = datasets_dir() / "HotpotQA" / "hotpot_dev_fullwiki_v1.json"
    data = []
    with open(data_path, "r") as f:
        data = json.load(f)

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

    for i, data in enumerate(selected_data):
        logging.info(f"Running data {i}: {data['question']}: {data['answer']}")
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
                HotpotQAPrompter(),
                HotpotQAParser(),
                {
                    "original": data["question"],
                    "ground_truth": data["answer"],
                    "current": "",
                    "phase": 0,
                    "method": method.__name__,
                },
            )
            executor.run()
            # try:
            #     executor.run()
            # except Exception as e:
            #     logging.error(f"Exception: {e}")
            path = os.path.join(
                results_folder,
                method.__name__,
                f"{i}.json",
            )
            executor.output_graph(path)
            budget -= lm.cost

    return orig_budget - budget

    pass


if __name__ == "__main__":
    budget = 30
    samples = [item for item in range(30)]
    approaches = [io_base]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S'
    )

    spent = run(samples, approaches, budget, "chatgpt")  # llama3-8b-ollama

    logging.info(f"Spent {spent} out of {budget} budget.")
