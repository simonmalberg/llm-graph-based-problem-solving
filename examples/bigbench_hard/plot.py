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
        result_directory=Path(__file__).parent / "results" / "llama3-8b-ollama_io-cot-cot_zeroshot-cot_sc-tot-plan_solve-plan_solve_plus_2024-07-10_11-52-49",
        config=Config(
            methods_order=["io", "cot", "cot_zeroshot", "cot_sc", "tot", "plan_solve", "plan_solve_plus"],
            methods_labels=["IO", "CoT", "CoT Zero", "CoT SC", "ToT", "P&S", "P&S+"],
            y_lower=0,
            y_upper=1.2,
            cost_upper=1,
            cost_in_percent=True,
            display_solved=True,
            annotation_offset=-0.1,
            display_left_ylabel=True,
            display_right_ylabel=True,
            left_ylabel="Score (1 or 0); the higher the better",
            right_ylabel="Percentage of total token count",
            figsize=(7.5, 5),
            fig_fontsize=12,
            # title="correct results out of 20",
            aggregate_subtasks=True,
            )
        )
    doc_plotter.plot_results()
    # doc_plotter = GSM8KPlotter(
    #     # result_directory=Path(__file__).parent / "results" / "llama3-8b-ollama_io-cot_2024-06-12_14-50-52",
    #     result_directory=Path(__file__).parent / "results" / "llama3-8b-ollama_io-cot-cot_zeroshot-cot_sc-tot-plan_solve-plan_solve_plus_2024-07-10_11-52-49" / "dyck_languages",
    #     config=Config(
    #         methods_order=["io", "cot", "cot_zeroshot", "cot_sc", "tot", "plan_solve", "plan_solve_plus"],
    #         methods_labels=["IO", "CoT", "CoT Zero", "CoT SC", "ToT", "P&S", "P&S+"],
    #         y_lower=0,
    #         y_upper=1.2,
    #         cost_upper=1,
    #         cost_in_percent=True,
    #         display_solved=True,
    #         annotation_offset=-0.1,
    #         display_left_ylabel=True,
    #         display_right_ylabel=True,
    #         left_ylabel="Score (1 or 0); the higher the better",
    #         right_ylabel="Percentage of total token count",
    #         figsize=(7.5, 5),
    #         fig_fontsize=12,
    #         title="correct results out of 20",
    #         )
    #     )
    # doc_plotter.plot_results()
