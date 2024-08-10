import re
from typing import List
import bm25s
from graph_of_thoughts.language_models.abstract_language_model import AbstractLanguageModel
from graph_of_thoughts.operations.operations import Operation, OperationType
from graph_of_thoughts.operations.thought import Thought
from graph_of_thoughts.parser.parser import Parser
from graph_of_thoughts.prompter.prompter import Prompter
from graph_of_thoughts.operations.probtree_tree import Node

class GraphBuildError(Exception):
    pass

def get_list_of_reference_ids(question: str) -> List[int]:
    pattern = r'<(\d+)>'
    matches = re.findall(pattern, question)
    integers = list(map(int, matches))
    print(integers)
    return integers

class ProbtreeReasoning(Operation):

    operation_type: OperationType = OperationType.probtree_reason

    def __init__(self, bm25_retriever_save_dir: str) -> None:
        super().__init__()
        self.thoughts: List[Thought] = []
        self.retriever = bm25s.BM25.load(bm25_retriever_save_dir, load_corpus=True, mmap=True)

    def build_tree(self) -> Node:
        try:
            node_for_question = {}
            tree = None
            for question, subquestions_and_logprob in self.dict_from_understanding.items():
                subquestions = subquestions_and_logprob[0]
                logprob = subquestions_and_logprob[1]
                if not (node := node_for_question.get(question)):
                    node = Node(question=question, logprob=logprob, retriever=self.retriever)
                    if tree is None:
                        tree = node
                if node.logprob is None:
                    node.logprob = logprob
                node_for_question[question] = node
                for subquestion in subquestions:
                    subnode = Node(question=subquestion, retriever=self.retriever)
                    reference_ids = get_list_of_reference_ids(subquestion)
                    if len(reference_ids) > 0:
                        for reference_id in reference_ids:
                            reference_node = node_for_question[subquestions[reference_id-1]]
                            subnode.references[reference_id] = reference_node
                    node.add_child(subnode)
                    node_for_question[subquestion] = subnode
            return tree
        except Exception as e:
            raise GraphBuildError(e)
        
    def find_next_executable_node(self):
        # A helper function to recursively perform DFS
        def dfs(current_node: Node):
            # If the current node is not executed and executable, return it
            if not current_node.executed and current_node.executable:
                return current_node
            
            # Recursively visit children
            for child in current_node.children:
                if (next_node_to_execute := dfs(child)):
                    return next_node_to_execute
            
            # If no executable node is found, return None
            return None
        
        return dfs(self.tree)
    
    def get_thoughts(self) -> List[Thought]:
        return self.thoughts
    
    def _execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs):
        base_state = self.get_previous_thoughts()[0].state
        self.dict_from_understanding = base_state["current"]
        self.tree = self.build_tree()
        while not self.tree.executed:
            node = self.find_next_executable_node()
            node.execute(lm, prompter, parser, **kwargs)
        new_state = base_state.copy()
        new_state["tree"] = self.tree.to_dict()
        new_state["current"] = self.tree.answers.final_answer.text
        new_state["phase"] += 1
        self.thoughts.append(Thought({**base_state, **new_state}))
