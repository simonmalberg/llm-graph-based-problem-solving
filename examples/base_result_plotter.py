# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach
# contributions: Robert Gerstenberger, Felix Fricke (TU Munich)

from dataclasses import dataclass
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class Config:
    methods_order: List[str]
    methods_labels: List[str]
    y_lower: float
    y_upper: float
    cost_upper: float
    display_left_ylabel: bool = True
    left_ylabel: str = "Score"
    figsize: tuple = (2.5, 5)
    title: str = None
    fig_fontsize: int = 12
    cost_in_percent: bool = False
    display_right_ylabel: bool = True
    right_ylabel: str = "Total Cost ($)"
    display_solved: bool = True
    annotation_offset: float = 1.0
    aggregate_subtasks: bool = False
    plot_only_accuracy: bool = False


class BaseResultPlotter(ABC):
    def __init__(self, result_directory: Path, config: Config):
        # check if path exists:
        if not os.path.exists(result_directory):
            raise FileNotFoundError(f"Path {result_directory} not found.")
        self.result_directory = result_directory
        self.config = config


    def get_complete_results(self, aggregate_subtasks: bool = False):
        results_complete = {}
        if aggregate_subtasks:
            for taskfolder_name in os.listdir(self.result_directory):
                taskfolder_path = os.path.join(self.result_directory, taskfolder_name)
                if os.path.isdir(taskfolder_path):
                    for folder_name in os.listdir(taskfolder_path):
                        folder_path = os.path.join(taskfolder_path, folder_name)
                        if os.path.isdir(folder_path):
                            if folder_name not in results_complete:
                                results_complete[folder_name] = []
                            for file_name in os.listdir(folder_path):
                                if file_name.endswith(".json"):
                                    file_path = os.path.join(folder_path, file_name)
                                    with open(file_path, "r") as f:
                                        data = json.load(f)
                                        results_complete[folder_name].append(
                                            {"key": f"{taskfolder_name}_{int(file_name.split(".")[0])}", "data": data}
                                        )
                        for key in results_complete.keys():
                            results_complete[key] = sorted(
                                results_complete[key], key=lambda x: x["key"]
                            )
        for folder_name in os.listdir(self.result_directory):
            folder_path = os.path.join(self.result_directory, folder_name)
            if os.path.isdir(folder_path):
                results_complete[folder_name] = []
                for file_name in os.listdir(folder_path):
                    if file_name.endswith(".json"):
                        file_path = os.path.join(folder_path, file_name)
                        with open(file_path, "r") as f:
                            data = json.load(f)
                            results_complete[folder_name].append(
                                {"key": int(file_name.split(".")[0]), "data": data}
                            )
            for key in results_complete.keys():
                results_complete[key] = sorted(
                    results_complete[key], key=lambda x: x["key"]
                )
        return results_complete

    def get_final_scores(self, results_complete):
        scores = {}
        for method in results_complete.keys():
            scores[method] = []
            for result in results_complete[method]:
                score, solved, cost, prompt_tokens, completion_tokens = self.process_result(result)
                scores[method].append(
                    [result["key"], score, solved, prompt_tokens, completion_tokens, cost]
                )
            scores[method] = sorted(scores[method], key=lambda x: x[0])
        # change cost to be relative to the maximum cost
        if self.config.cost_in_percent:
            max_overall_cost = max([x[5] for method in scores.keys() for x in scores[method]])
            for method in scores.keys():
                for i in range(len(scores[method])):
                    scores[method][i][5] = scores[method][i][5] / max_overall_cost / len(scores[method])
        return scores

    @abstractmethod
    def process_result(self, result):
        pass

    def calculate_confidence_interval_for_binary_data(self, total, solved):
        if total == 0:
            return 0
        p = solved / total
        Z_value = 1.96
        return Z_value * ((p * (1 - p) / total) ** 0.5)
    
    # def calculate_mean_and_confidence_interval_for_general_score(self, scores):
    #     if len(scores) == 0:
    #         return 0
    #     mean = sum(scores) / len(scores)
    #     variance = sum([(x - mean) ** 2 for x in scores]) / (len(scores) - 1)
    #     standard_error = (variance / len(scores)) ** 0.5
    #     confidence_interval = 1.96 * standard_error
    #     return mean, confidence_interval
    
    def bootstrap_confidence_interval(self, scores, num_samples=1000, confidence_level=0.95):
        n = len(scores)
        means = []
        for _ in range(num_samples):
            sample = np.random.choice(scores, size=n, replace=True)
            means.append(np.mean(sample))
        means = sorted(means)
        lower_bound = np.percentile(means, (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(means, (1 + confidence_level) / 2 * 100)
        return lower_bound, upper_bound

    def get_plotting_data(self):
        results_complete = self.get_complete_results(aggregate_subtasks=self.config.aggregate_subtasks)
        scores = self.get_final_scores(results_complete)
        results_plotting = {
            method: {
                "scores": [x[1] for x in scores[method]],
                "solved": sum([1 for x in scores[method] if x[2]]),
                "costs": [x[5] for x in scores[method]],
            }
            for method in scores.keys()
        }
        return results_plotting

    def plot_results(self):
        results = self.get_plotting_data()
        methods_order = [method for method in self.config.methods_order if method in results]
        scores_ordered = [
            [score for score in results[method]["scores"] if score != 1000]
            for method in methods_order
        ]
        total_costs = [sum(results[method]["costs"]) for method in methods_order]

        fig, ax = plt.subplots(dpi=150, figsize=self.config.figsize)
        positions = range(1, len(methods_order) + 1)
        fig_fontsize = self.config.fig_fontsize
        ax2_width = 0.25
        ax2_offset = 0.25

        latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|r|r|}\n\\hline\n"

        if self.config.plot_only_accuracy:
            latex_table += "Method & \multicolumn{1}{c|}{Accuracy} & \multicolumn{1}{c|}{Cost (\$)} \\\\\n\\hline\n"
            confidence_intervals = [    
                self.calculate_confidence_interval_for_binary_data(len(scores), sum(scores))
                for scores in scores_ordered
            ]
            means = [sum(scores) / len(scores) for scores in scores_ordered]
            ax.bar(positions, means, color="black", alpha=0.5, width=0.25)
            for i, (method, scores) in enumerate(zip(methods_order, scores_ordered)):
                score_entry = f"${means[i]*100:.1f}\% \\pm {confidence_intervals[i]*100:.1f}\%$"
                cost_entry = f"${total_costs[i]:.2f}$"
                latex_table += f"{self.config.methods_labels[i]} & {score_entry} & {cost_entry} \\\\\n\\hline\n"
                ax.errorbar(
                    positions[i],
                    sum(scores) / len(scores),
                    yerr=confidence_intervals[i],
                    fmt="",
                    color="black",
                    capsize=5,
                )
        else:
            latex_table += "Method & \multicolumn{1}{c|}{F1} & \multicolumn{1}{c|}{Cost (\$)} \\\\\n\\hline\n"
            bounds = [
                self.bootstrap_confidence_interval(scores)
                for scores in scores_ordered
            ]
            lower_bounds = [x[0] for x in bounds]
            upper_bounds = [x[1] for x in bounds]
            means = [sum(scores) / len(scores) for scores in scores_ordered]
            ax.bar(positions, means, color="black", alpha=0.5, width=0.25)
            for i, (method, scores) in enumerate(zip(methods_order, scores_ordered)):
                y_err = [[means[i] - lower_bounds[i]], [upper_bounds[i] - means[i]]]
                score_entry = f"${means[i]:.2f} \\pm ({means[i] - lower_bounds[i]:.2f}, {upper_bounds[i] - means[i]:.2f})$"
                cost_entry = f"${total_costs[i]:.2f}$"
                
                latex_table += f"{self.config.methods_labels[i]} & {score_entry} & {cost_entry} \\\\\n\\hline\n"
                ax.errorbar(
                    positions[i],
                    sum(scores) / len(scores),
                    yerr=y_err,
                    fmt="",
                    color="black",
                    capsize=5,
                )
            # yerr = [[mean_score - lower_bound], [upper_bound - mean_score]]:
            
            # ax.boxplot(scores_ordered, positions=positions, meanline=True, showmeans=True)
            # ax.set_ylim(self.config.y_lower, self.config.y_upper)
            # ax2_width = 1
            # ax2_offset = 0
        
        ax.set_ylim(self.config.y_lower, self.config.y_upper)

        ax.set_xticks(range(1, len(methods_order) + 1))
        ax.set_xticklabels(self.config.methods_labels, fontsize=fig_fontsize)

        if self.config.display_left_ylabel:
            ax.set_ylabel(self.config.left_ylabel, fontsize=fig_fontsize)

        if self.config.title:
            ax.set_title(self.config.title)
        else:
            ax.set_title(f"# correct results out of {len(scores_ordered[0])}")

        ax2 = ax.twinx()
        #add offset to positions:
        positions = [x + ax2_offset for x in positions]
        ax2.bar(positions, total_costs, alpha=0.5, color="blue", label="Total Cost ($)", width=ax2_width)
        ax2.yaxis.set_tick_params(colors="#1919ff", labelsize=fig_fontsize)
        if self.config.cost_upper > 0:
            ax2.set_ylim(0, self.config.cost_upper)
            if self.config.cost_in_percent:
                number_of_ticks = 11
                tick_interval = 0.1
            else:
                number_of_ticks = len(ax.get_yticks() + 1)
                tick_interval = self.config.cost_upper / number_of_ticks
            ax2_ticks = [tick_interval * i for i in range(number_of_ticks)]
            ax2.set_yticks(ax2_ticks)

        if self.config.display_right_ylabel:
            ax2.set_ylabel(self.config.right_ylabel, color="#1919ff", fontsize=fig_fontsize)

        if self.config.display_solved:
            annotation_height = self.config.y_upper + self.config.annotation_offset
            count = 1
            for method in methods_order:
                if method not in results:
                    continue
                solved = results[method]["solved"]
                ax.text(
                    count,
                    annotation_height,
                    f"{solved}",
                    ha="center",
                    va="bottom",
                    fontsize=fig_fontsize,
                )
                count += 1

        fig.savefig(f"{self.result_directory}.pdf", bbox_inches="tight")
        fig.clear()
        latex_table += "\\end{tabular}\n\\caption{Performance and Cost Summary}\n\\end{table}"
        with open(f"{self.result_directory}_table.tex", "w") as f:
            f.write(latex_table)

