# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

import json
import logging
from typing import List
from graph_of_thoughts.language_models import AbstractLanguageModel
from graph_of_thoughts.operations import GraphOfOperations, Thought
from graph_of_thoughts.prompter import Prompter
from graph_of_thoughts.parser import Parser

import uuid


class Controller:
    """
    Controller class to manage the execution flow of the Graph of Operations,
    generating the Graph Reasoning State.
    This involves language models, graph operations, prompting, and parsing.
    """

    def __init__(
        self,
        lm: AbstractLanguageModel,
        graph: GraphOfOperations,
        prompter: Prompter,
        parser: Parser,
        problem_parameters: dict,
    ) -> None:
        """
        Initialize the Controller instance with the language model,
        operations graph, prompter, parser, and problem parameters.

        :param lm: An instance of the AbstractLanguageModel.
        :type lm: AbstractLanguageModel
        :param graph: The Graph of Operations to be executed.
        :type graph: OperationsGraph
        :param prompter: An instance of the Prompter class, used to generate prompts.
        :type prompter: Prompter
        :param parser: An instance of the Parser class, used to parse responses.
        :type parser: Parser
        :param problem_parameters: Initial parameters/state of the problem.
        :type problem_parameters: dict
        """
        self.logger = logging.getLogger(self.__class__.__module__)
        self.lm = lm
        self.graph = graph
        self.prompter = prompter
        self.parser = parser
        self.problem_parameters = problem_parameters
        self.run_executed = False

    def run(self) -> None:
        """
        Run the controller and execute the operations from the Graph of
        Operations based on their readiness.
        Ensures the program is in a valid state before execution.
        :raises AssertionError: If the Graph of Operation has no roots.
        :raises AssertionError: If the successor of an operation is not in the Graph of Operations.
        """
        self.logger.debug("Checking that the program is in a valid state")
        assert self.graph.roots is not None, "The operations graph has no root"
        self.logger.debug("The program is in a valid state")

        execution_queue = [
            operation
            for operation in self.graph.operations
            if operation.can_be_executed()
        ]

        while len(execution_queue) > 0:
            current_operation = execution_queue.pop(0)
            self.logger.info("===========================================================================")
            self.logger.info("Executing operation %s", current_operation.display_name)
            self.logger.info("===========================================================================")
            current_operation.execute(
                self.lm, self.prompter, self.parser, **self.problem_parameters
            )
            self.logger.info("Operation %s executed", current_operation.operation_type)
            for operation in current_operation.successors:
                assert (
                    operation in self.graph.operations
                ), "The successor of an operation is not in the operations graph"
                if operation.can_be_executed():
                    execution_queue.append(operation)
        self.logger.info("All operations executed")
        self.run_executed = True

    def get_final_thoughts(self) -> List[List[Thought]]:
        """
        Retrieve the final thoughts after all operations have been executed.

        :return: List of thoughts for each operation in the graph's leaves.
        :rtype: List[List[Thought]]
        :raises AssertionError: If the `run` method hasn't been executed yet.
        """
        assert self.run_executed, "The run method has not been executed"
        return [operation.get_thoughts() for operation in self.graph.leaves]

    def output_graph(self, path: str) -> None:
        """
        Serialize the state and results of the operations graph to a JSON file.

        :param path: The path to the output file.
        :type path: str
        """
        output = []
        for operation in self.graph.operations:
            operation_serialized = {
                "operation": operation.operation_type.name,
                "thoughts": [thought.state for thought in operation.get_thoughts()],
            }
            if any([thought.scored for thought in operation.get_thoughts()]):
                operation_serialized["scored"] = [
                    thought.scored for thought in operation.get_thoughts()
                ]
                operation_serialized["scores"] = [
                    thought.score for thought in operation.get_thoughts()
                ]
            if any([thought.validated for thought in operation.get_thoughts()]):
                operation_serialized["validated"] = [
                    thought.validated for thought in operation.get_thoughts()
                ]
                operation_serialized["validity"] = [
                    thought.valid for thought in operation.get_thoughts()
                ]
            if any(
                [
                    thought.compared_to_ground_truth
                    for thought in operation.get_thoughts()
                ]
            ):
                operation_serialized["compared_to_ground_truth"] = [
                    thought.compared_to_ground_truth
                    for thought in operation.get_thoughts()
                ]
                operation_serialized["problem_solved"] = [
                    thought.solved for thought in operation.get_thoughts()
                ]
            output.append(operation_serialized)

        output.append(
            {
                "prompt_tokens": self.lm.prompt_tokens,
                "completion_tokens": self.lm.completion_tokens,
                "cost": self.lm.cost,
            }
        )

        with open(path, "w") as file:
            file.write(json.dumps(output, indent=2))


    def generate_json_canvas(self, path: str):
        canvas_dict = {
            "nodes": [],
            "edges": []
        }
        for i, operation in enumerate(self.graph.operations):
            node_operation = {
                "id": str(uuid.uuid4()),
                "type": "text",
                "x": 0,
                "y": (2*i)*600,
                "width": 640,
                "height": 80,
                "text": f"### {operation.display_name}"
            }
            canvas_dict["nodes"].append(node_operation)
            thought_count_middle = int(len(operation.get_thoughts())/2)
            for j, thought in enumerate(operation.get_thoughts()):
                thought_string = f"### Thoughts\n id: {thought.id}\n"
                if thought.scored:
                    thought_string = thought_string + f"score: {thought.score}\n"
                for key in thought.state.keys():
                    thought_string = thought_string + f"{key}:\n ```\n{thought.state[key]}\n```\n\n"

                node_thought = {
                    "id": str(uuid.uuid4()),
                    "type": "text",
                    "x": (j-thought_count_middle)*600,
                    "y": ((2*i)*600 + 300),
                    "width": 560,
                    "height": 720,
                    "text": thought_string
                }
                canvas_dict["nodes"].append(node_thought)
                edge_thought = {
                    "id": str(uuid.uuid4()),
                    "fromNode": node_operation["id"],
                    "toNode": node_thought["id"],
                    "fromSide": "bottom",
                    "toSide": "top",
                    "toEnd": "arrow",
                }
                canvas_dict["edges"].append(edge_thought)
        with open(path, "w") as file:
            file.write(json.dumps(canvas_dict, indent=2))

