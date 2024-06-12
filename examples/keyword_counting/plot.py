from pathlib import Path
from examples.base_result_plotter import BaseResultPlotter, Config

class KeywordCountingPlotter(BaseResultPlotter):

    def process_result(self, result):
        score = 100
        solved = False
        cost = 1
        prompt_tokens = 0
        completion_tokens = 0
        for op in result["data"]:
            if "operation" in op and op["operation"] == "ground_truth_evaluator":
                try:
                    score = min(op["scores"])
                    solved = any(op["problem_solved"])
                except:
                    continue
            if "cost" in op:
                cost = op["cost"]
                prompt_tokens = op["prompt_tokens"]
                completion_tokens = op["completion_tokens"]
        return score, solved, cost, prompt_tokens, completion_tokens


if __name__ == "__main__":
    plotter = KeywordCountingPlotter(
        result_directory=Path(__file__).parent / "results" / "ADD YOUR RESULT DIRECTORY HERE",
        config=Config(
            methods_order=["io", "cot", "tot", "tot2", "got4", "got8", "gotx"],
            methods_labels=["IO", "CoT", "ToT", "ToT2", "GoT4", "GoT8", "GoTx"],
            y_lower=0,
            y_upper=35,
            cost_upper=9,
            display_solved=True,
            annotation_offset=-0.3,
            display_left_ylabel=True,
            display_right_ylabel=True,
            left_ylabel="Number of errors; the lower the better",
            right_ylabel="Total Cost ($); the lower the better",
            figsize=(3.75, 4),
            fig_fontsize=12,
            title="Keyword Counting"
        )
        )
    plotter.plot_results()
