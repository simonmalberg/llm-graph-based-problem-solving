from pathlib import Path
from examples.base_result_plotter import BaseResultPlotter, Config

class BigbenchHardPlotter(BaseResultPlotter):

    def process_result(self, result):
        score = 0
        solved = False
        cost = 1
        prompt_tokens = 0
        completion_tokens = 0
        for op in result["data"]:
            if "operation" in op and op["operation"] == "ground_truth_evaluator":
                try:
                    solved = any(op["problem_solved"])
                    score = 0 if not solved else 1
                except:
                    continue
            if "cost" in op:
                # prompt_tokens = op["prompt_tokens"]
                # completion_tokens = op["completion_tokens"]
                # cost = prompt_tokens + completion_tokens
                cost = op["cost"]
        return score, solved, cost, prompt_tokens, completion_tokens


if __name__ == "__main__":
    doc_plotter = BigbenchHardPlotter(
        # result_directory=Path(__file__).parent / "results" / "llama3-8b-ollama_io-cot_2024-06-12_14-50-52",
        result_directory=Path(__file__).parent / "results" / "chatgpt_io-io_zs-cot-cot_zeroshot-cot_sc-tot-plan_solve-plan_solve_plus-got_2024-08-10_19-03-03",
        config=Config(
            methods_order=["io_zs", "plan_solve", "plan_solve_plus", "cot_zeroshot", "io", "cot", "cot_sc", "tot", "got"],
            methods_labels=["IO-zs", "PS", "PS+", "CoT-zs", "IO", "CoT", "CoT-SC", "ToT", "GoT"],
            y_lower=0,
            y_upper=0.5,
            cost_upper=12,
            cost_in_percent=False,
            display_solved=True,
            annotation_offset=-0.035,
            display_left_ylabel=True,
            display_right_ylabel=True,
            left_ylabel="Accuracy",
            right_ylabel="Total cost per method in USD",
            figsize=(7.5, 4),
            fig_fontsize=12,
            # title="correct results out of 20",
            aggregate_subtasks=True,
            plot_only_accuracy=True,
            )
        )
    doc_plotter.plot_results()
