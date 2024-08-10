from dataclasses import asdict, dataclass
from typing import Any, Dict, List
import re
import bm25s

from graph_of_thoughts.language_models.abstract_language_model import AbstractLanguageModel
from graph_of_thoughts.prompter.prompter import Prompter
from graph_of_thoughts.parser.parser import Parser


def parse_bm25_documents(bm25_result: bm25s.Results) -> str:
    context_str = ""
    documents = bm25_result.documents[0]
    for i, doc in enumerate(documents):
        context_str += f"\n#{i+1} Wikipedia Title: "
        context_str += doc["title"]
        context_str += "\nText: "
        context_str += "".join(doc["text"])
    return context_str


def postprocess_answer(response):
    """
    Code adapted from the ProbTree repository (https://github.com/THU-KEG/ProbTree)
    """
    tokens = [per_token.token for per_token in response.choices[0].logprobs.content]
    token_logprobs = [per_token.logprob for per_token in response.choices[0].logprobs.content]
    cot = response.choices[0].message.content.strip()
    if len(token_logprobs) == 0:
        return 'ERROR: empty output', -100, cot
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

    def __init__(self, retriever: bm25s.BM25, question: str, logprob: float = None) -> None:
        self.question: str = question
        self.children: List[Node] = []
        self.references: Dict[int, Node] = dict()
        self.logprob: float = logprob
        self.answers: Answers = Answers()
        self.retriever = retriever

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
    
    def get_discounted_ca_logprob(self, ca_answer_confidence: float):
        num_children = len(self.children)
        factor = 1.0 / (num_children + 2)
        question_decomposition_score = self.logprob
        children_max_scores = [child.answers.final_answer.logprob for child in self.children]
        return factor * (question_decomposition_score + ca_answer_confidence + sum(children_max_scores))

        
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
        # open book multihop
        with open('graph_of_thoughts/prompter/prompts/probtree_ob_multihop.txt', 'r') as file:
            prompt_ob_multihop = file.read()
        question_tokenized = bm25s.tokenize(self.question_with_reference_answers, stopwords="en")
        context_raw = self.retriever.retrieve(question_tokenized, k=5)
        context = parse_bm25_documents(context_raw)
        prompt_ob_multihop = prompt_ob_multihop.format(question=self.question_with_reference_answers, context=context)
        response = lm.query(prompt_ob_multihop, num_responses=1, logprobs=True)
        answer_ob_multihop = postprocess_answer(response)
        self.answers.open_book = Answer(text=answer_ob_multihop[0], text_with_reasoning=answer_ob_multihop[2], logprob=answer_ob_multihop[1])
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
            discounted_ca_logprob = self.get_discounted_ca_logprob(ca_answer_confidence=answer_ca[1])
            self.answers.child_aggregating = Answer(text=answer_ca[0], text_with_reasoning=answer_ca[2], logprob=discounted_ca_logprob)
