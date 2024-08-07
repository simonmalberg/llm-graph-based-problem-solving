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
                prompt_tokens = op["prompt_tokens"]
                completion_tokens = op["completion_tokens"]
                cost = prompt_tokens + completion_tokens
        return score, solved, cost, prompt_tokens, completion_tokens


if __name__ == "__main__":
    doc_plotter = GSM8KPlotter(
        # result_directory=Path(__file__).parent / "results" / "llama3-8b-ollama_io-cot_2024-06-12_14-50-52",
        result_directory=Path(__file__).parent / "results" / "gpt-3.5-2024-07-10",
        config=Config(
            
            methods_order=["io","cot", "cot_zeroshot","cot_sc", "plan_solve","plan_solve_plus","tot_base","tot_style"],
            methods_labels=["IO", "CoT-SC"],
            y_lower=0,
            y_upper=2,
            cost_upper=1,
            cost_in_percent=True,
            display_solved=True,
            annotation_offset=-0.1,
            display_left_ylabel=True,
            display_right_ylabel=True,
            left_ylabel="Score (1 or 0); the higher the better",
            right_ylabel="Percentage of total token count",
            figsize=(10, 5),
            fig_fontsize=14,
        )
        )
    doc_plotter.plot_results()
