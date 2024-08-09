from dataclasses import asdict, dataclass
from typing import Dict, List
import re

from graph_of_thoughts.language_models.abstract_language_model import AbstractLanguageModel
from graph_of_thoughts.prompter.prompter import Prompter
from graph_of_thoughts.parser.parser import Parser



def postprocess_answer(response):
    tokens = [per_token.token for per_token in response.choices[0].logprobs.content]
    token_logprobs = [per_token.logprob for per_token in response.choices[0].logprobs.content]
    cot = response.choices[0].message.content.strip()
    if len(token_logprobs) == 0:
        return 'ERROR: empty output', -100, cot
    # if "Unknown" in cot:
    #     return "Unknow", sum(token_logprobs) / len(token_logprobs), cot
    pos = 0
    for idx, token in enumerate(tokens):
        if token.strip() == 'So' and idx + 1 <= len(tokens) and tokens[idx + 1].strip() == 'the' and idx + 2 <= len(tokens) and tokens[idx + 2].strip() == 'answer' and idx + 3 <= len(tokens) and tokens[idx + 3].strip() == 'is' and idx + 4 <= len(tokens) and tokens[idx + 4].strip() == ':':
            pos = idx
            break
    if tokens[-1] == '.':
        answer_logprobs = token_logprobs[pos+5:-1]
        answer = cot.split('So the answer is: ')[-1][:-1]
    else:
        answer_logprobs = token_logprobs[pos+5:]
        answer = cot.split('So the answer is: ')[-1]
    cot_process = cot.split('So the answer is: ')[0].strip()
    cot_process_logprobs = token_logprobs[:pos]
    if len(cot_process_logprobs) == 0:
        cot_process_logprob = -100
    else:
        cot_process_logprob = sum(cot_process_logprobs) / len(cot_process_logprobs)
    return answer, cot_process_logprob, cot

@dataclass
class Answer:
    text: str
    text_with_reasoning: str
    logprob: float

    def to_dict(self):
        return asdict(self)

@dataclass
class Answers:
    closed_book: Answer = None
    open_book: Answer = None
    child_aggregating: Answer = None

    def to_dict(self):
        return {
            "closed_book": self.closed_book.to_dict() if self.closed_book is not None else None,
            "open_book": self.open_book.to_dict() if self.open_book is not None else None,
            "child_aggregating": self.child_aggregating.to_dict() if self.child_aggregating is not None else None
        }

    @property
    def is_empty(self):
        return self.closed_book is None and self.open_book is None and self.child_aggregating is None
    
    @property
    def final_answer(self) -> Answer:
        processed_answers = [answer for answer in [self.closed_book, self.open_book, self.child_aggregating] if answer is not None]
        return max(processed_answers, key=lambda x: x.logprob)


class Node:

    def __init__(self, question: str, logprob: float = None) -> None:
        self.question: str = question
        self.children: List[Node] = []
        self.references: Dict[int, Node] = dict()
        self.logprob: float = logprob
        self.answers: Answers = Answers()

    def to_dict(self):
        return {
            "question": self.question,
            "children": [child.to_dict() for child in self.children],
            # "references": {key: reference.to_dict() for key, reference in self.references.items()},
            "logprob": self.logprob,
            "answers": self.answers.to_dict()
        }

    def add_child(self, child: 'Node'):
        self.children.append(child)
    
    def set_logprob(self, logprob: float):
        self.logprob = logprob

    @property
    def question_with_reference_answers(self) -> str:
        pattern = r'<(\d+)>'
        matches = re.findall(pattern, self.question)
        question_with_reference_answers = self.question
        for match in matches:
            match_int = int(match)
            question_with_reference_answers = question_with_reference_answers.replace(f'<{match_int}>', self.references[match_int].answers.final_answer.text)
        return question_with_reference_answers

    @property
    def is_leaf(self):
        return len(self.children) == 0 and len(self.references) == 0
    
    @property
    def executed(self):
        return not self.answers.is_empty
    
    @property
    def executable(self):
        for child in self.children + list(self.references.values()):
            if child.answers.is_empty:
                return False
        return True
        
    def execute(self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs):
        if not self.executable:
            raise ValueError("Node is not executable")
        # closed book
        with open('graph_of_thoughts/prompter/prompts/probtree_cb.txt', 'r') as file:
            prompt_cb = file.read()
        prompt_cb = prompt_cb.format(question=self.question_with_reference_answers)
        response = lm.query(prompt_cb, num_responses=1, logprobs=True)
        answer_cb = postprocess_answer(response)
        self.answers.closed_book = Answer(text=answer_cb[0], text_with_reasoning=answer_cb[2], logprob=answer_cb[1])
        # # open book multihop
        # with open('graph_of_thoughts/prompter/prompts/probtree_ob_multihop.txt', 'r') as file:
        #     prompt_ob_multihop = file.read()
        # # open book singlehop
        # with open('graph_of_thoughts/prompter/prompts/probtree_ob_singlehop.txt', 'r') as file:
        #     prompt_ob_singlehop = file.read()
        # child aggregating
        if len(self.children) > 0:
            with open('graph_of_thoughts/prompter/prompts/probtree_ca.txt', 'r') as file:
                prompt_ca = file.read()
            context = ""
            for child in self.children:
                context += child.question_with_reference_answers + " "
                context += child.answers.final_answer.text + "\n"
            prompt_ca = prompt_ca.format(question=self.question_with_reference_answers, context=context)
            response = lm.query(prompt_ca, num_responses=1, logprobs=True)
            answer_ca = postprocess_answer(response)
            self.answers.child_aggregating = Answer(text=answer_ca[0], text_with_reasoning=answer_ca[2], logprob=answer_ca[1])
        
    


    

# if __name__ == "__main__":
#     test_dict = {
#         "Who was known by his stage name Aladin and helped organizations improve their performance as a consultant?": [
#         [
#             "What was the stage name of the person known as Aladin?",
#             "Who helped organizations improve their performance as a consultant?"
#         ],
#         -0.15340366200627834
#         ],
#         "What was the stage name of the person known as Aladin?": [
#         [
#             "Who was known by the stage name Aladin?",
#             "What was the stage name of <1>?"
#         ],
#         -0.06493883921907692
#         ]
#     }
#     operations_graph = operations.GraphOfOperations()
#     lm = language_models.ChatGPT(
#                 os.path.join(
#                     os.path.dirname(__file__),
#                     "../../language_models/config.json",
#                 ),
#                 model_name="chatgpt",
#                 cache=True,
#             )
#     exection_graph = ProbtreeExecutionGraph()
#     operations_graph.add_operation(exection_graph)
#     executor = controller.Controller(
#                 lm,
#                 operations_graph,
#                 HotpotQAPrompter(),
#                 HotpotQAParser(),
#                 {
#                     "original": test_dict,
#                     "current": "",
#                     "phase": 0,
#                 },
#             )
#     pass