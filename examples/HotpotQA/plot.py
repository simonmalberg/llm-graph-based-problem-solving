from pathlib import Path
from examples.base_result_plotter import BaseResultPlotter, Config

class DocMergePlotter(BaseResultPlotter):

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
        return score, solved, cost, prompt_tokens, completion_tokens


if __name__ == "__main__":
    plotter = DocMergePlotter(
        result_directory=Path(__file__).parent / "results" / "chatgpt_probtree_2024-08-10_13-44-46",
        config=Config(
            methods_order=["probtree"],
            methods_labels=["ProbTree"],
            y_lower=0,
            y_upper=1,
            cost_upper=1.2,
            display_solved=True,
            annotation_offset=-0.1,
            display_left_ylabel=True,
            display_right_ylabel=True,
            left_ylabel="F1 Score",
            right_ylabel="Total Cost ($); the lower the better",
            figsize=(3.75, 5),
            fig_fontsize=12,
        )
        )
    plotter.plot_results()
