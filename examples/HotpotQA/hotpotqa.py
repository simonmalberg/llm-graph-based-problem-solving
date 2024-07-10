import os
import logging
import datetime
import json
from pathlib import Path
from typing import List, Callable
from graph_of_thoughts import controller, language_models, operations

try:
    from .src.prompter import HotpotQAPrompter
    from .src.parser import HotpotQAParser
    from .src import utils
except ImportError:
    from src.prompter import HotpotQAPrompter
    from src.parser import HotpotQAParser
    from src import utils



def run(
        data_ids: List[int],
        methods: List[Callable[[], operations.GraphOfOperations]],
        budget: float,
        lm_name: str,
    ) -> float:

    pass


if __name__ == "__main__":
    budget = 30
    samples = [item for item in range(5)]
    approaches = [io, cot, cotsc, plan_and_solve]

    logging.basicConfig(level=logging.INFO)

    spent = run(samples, approaches, budget, "llama3-8b-ollama")

    logging.info(f"Spent {spent} out of {budget} budget.")
    