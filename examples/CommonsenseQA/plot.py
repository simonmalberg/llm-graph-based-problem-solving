from pathlib import Path
from examples.base_result_plotter import BaseResultPlotter, Config


class GSM8KPlotter(BaseResultPlotter):

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
                cost = op["cost"]
                completion_tokens = op["completion_tokens"]
                prompt_tokens = op["prompt_tokens"]
                # cost = prompt_tokens + completion_tokens
        return score, solved, cost, prompt_tokens, completion_tokens


if __name__ == "__main__":

    runs_to_plot = ["chatgpt_final", "gpt4_turbo_final", "replicate_final"]
    cost_upper = [60, 360, 12]

    run = "replicate_final"
    doc_plotter = GSM8KPlotter(
        # result_directory=Path(__file__).parent / "results" / "llama3-8b-ollama_io-cot_2024-06-12_14-50-52",
        result_directory=Path(__file__).parent / "results" / run,
        config=Config(

            methods_order=["plan_solve", "plan_solve_plus", "cot_zeroshot", "io", "cot", "cot_sc", "tot_base"],
            methods_labels=["PS", "PS+", "CoT-zs", "IO", "CoT", "CoT-SC", "ToT"],
            y_lower=0,
            y_upper=1,
            cost_upper=12,
            display_solved=True,
            annotation_offset=-0.1,
            display_left_ylabel=True,
            display_right_ylabel=True,
            left_ylabel="Accuracy",
            right_ylabel="Total cost per method in USD",
            figsize=(10, 5),
            fig_fontsize=14,
            plot_only_accuracy=True
        )
    )
    doc_plotter.plot_results()
