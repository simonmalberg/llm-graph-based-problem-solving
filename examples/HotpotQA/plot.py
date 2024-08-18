from pathlib import Path
from examples.base_result_plotter import BaseResultPlotter, Config

class HotpotQAPlotter(BaseResultPlotter):

    def process_result(self, result):
        score = 0
        solved = False
        cost = 1
        prompt_tokens = 0
        completion_tokens = 0
        for op in reversed(result["data"]):
            if "cost" in op:
                cost = op["cost"]
                prompt_tokens = op["prompt_tokens"]
                completion_tokens = op["completion_tokens"]
            if "operation" in op and op["operation"] == "score":
                try:
                    score = max(op["scores"])
                    break
                except:
                    continue
            if "operation" in op and op["operation"] == "ground_truth_evaluator":
                try:
                    solved = any(op["problem_solved"])
                    score = 0 if not solved else 1
                except:
                    continue
        return score, solved, cost, prompt_tokens, completion_tokens


if __name__ == "__main__":
    plotter = HotpotQAPlotter(
        result_directory=Path(__file__).parent / "results" / "chatgpt_final",
        config=Config(
            methods_order=["io_closedbook", "io_base", "io_zs", "io", "plan_solve_basic", "plan_solve_plus", "cot_zeroshot", "cot", "cot_sc_1", "cot_sc_2", "tot", "probtree"],
            methods_labels=["IO-CB", "IO-Base", "IO-zs", "IO", "PS", "PS+", "CoT-zs", "CoT", "CoT-SC1", "CoT-SC2", "ToT", "ProbTree"],
            # methods_order=["probtree"],
            # methods_labels=["ProbTree"],
            y_lower=0,
            y_upper=1.0,
            cost_upper=240,
            display_solved=True,
            annotation_offset=-0.05,
            display_left_ylabel=True,
            display_right_ylabel=True,
            left_ylabel="F1 Score",
            right_ylabel="Total Cost ($); the lower the better",
            figsize=(11.25, 5),
            fig_fontsize=12,
        )
        )
    plotter.plot_results()
