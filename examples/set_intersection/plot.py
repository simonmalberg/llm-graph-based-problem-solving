from examples.base_result_plotter import BaseResultPlotter, Config
from pathlib import Path

class SetIntersectionPlotter(BaseResultPlotter):
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
    set_plotter = SetIntersectionPlotter(
        result_directory=Path(__file__).parent / "results" / "chatgpt_io-cot-tot-tot2-got_2024-06-12_15-40-56",
        config=Config(
            methods_order=["io", "cot", "tot", "tot2", "got"],
            methods_labels=["IO", "CoT", "ToT", "ToT2", "GoT"],
            y_lower=0,
            y_upper=32,
            cost_upper=0.0,
            display_solved=True,
            annotation_offset=-1.5,
            display_left_ylabel=True,
            display_right_ylabel=True,
            left_ylabel="#incorrect elements; the lower the better",
            right_ylabel="Total Cost ($); the lower the better",
            title="32 elements",
            figsize=(2.5, 5),
            fig_fontsize=12,
            )
        )
    set_plotter.plot_results()