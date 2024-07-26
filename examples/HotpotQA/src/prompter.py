import logging
from typing import Dict, List

from examples.HotpotQA.src import utils
from graph_of_thoughts import prompter


class HotpotQAPrompter(prompter.Prompter):
    """
    HotpotQAPrompter provides the generation of prompts specific to the hotpotqa example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    io_prompt_get_keywords = """\
<Instruction>Give me a list of keywords for a wikipedia lookup to be able to answer this question. Give the keywords in the following format: <Keywords>["keyword1", "keyword2"]</Keywords>.</Instruction>
<Question>{input}</Question>
Output:
"""

    io_prompt_answer_question = """\
<Instruction>Answer the question using the provided context. Only output the final answer directly without any other text.
<Examples>
{examples}
</Examples>
</Instruction>
<Context>{context}</Context>
<Question>{question}</Question>
Output:
"""
    tree_generation_examples = """\
Q: Jeremy Theobald and Christopher Nolan share what profession?
A: {"Jeremy Theobald and Christopher Nolan share what profession?": ["What is Jeremy Theobald's profession?", "What is Christopher Nolan's profession?"]}.
Q: How many episodes were in the South Korean television series in which Ryu Hye−young played Bo−ra?
A: {"How many episodes were in the South Korean television series in which Ryu Hye−young played Bo−ra?": ["In which South Korean television series Ryu Hye−young played Bo−ra?", "How many episodes were <1>?"]}.
Q: Vertical Limit stars which actor who also played astronaut Alan Shepard in "The Right Stuff"?
A: {"Vertical Limit stars which actor who also played astronaut Alan Shepard in \"The Right Stuff\"?": ["Vertical Limit stars which actor?", "Which actor played astronaut Alan Shepard in \"The Right Stuff\"?"]}.
Q: What was the 2014 population of the city where Lake Wales Medical Center is located?
A: {"What was the 2014 population of the city where Lake Wales Medical Center is located?": ["Which city was Lake Wales Medical Center located in?", "What was the 2014 population of <1>?"]}.
Q: Who was born first? Jan de Bont or Raoul Walsh?
A: {"Who was born first? Jan de Bont or Raoul Walsh?": ["When was Jan de Bont born?", "When was Raoul Walsh born?"]}.
Q: In what country was Lost Gravity manufactured?
A: {"In what country was Lost Gravity manufactured?": ["Which company was Lost Gravity manufactured?", "Which country is <1> in?"]}.
Q: Which of the following had a debut album entitled "We Have an Emergency": Hot Hot Heat or The Operation M.D.?
A: {"Which of the following had a debut album entitled \"We Have an Emergency\": Hot Hot Heat or The Operation M.D.?": ["What is the debut album of the band Hot Hot Heat?", "What is the debut album of the band The Operation M.D.?"]}.
Q: In which country did this Australian who was detained in Guantanamo Bay detention camp and published "Guantanamo: My Journey" receive para−military training?
A: {"In which country did this Australian who was detained in Guantanamo Bay detention camp and published \"Guantanamo: My Journey\" receive para−military training?": ["Which Australian was detained in Guantanamo Bay detention camp and published \"Guantanamo: My Journey\"?", "In which country did <1> receive para−military training?"]}.
Q: Does The Border Surrender or Unsane have more members?
A: {"Does The Border Surrender or Unsane have more members?": ["How many members does The Border Surrender have?", "How many members does Unsane have?"]}.
Q: James Paris Lee is best known for investing the Lee−Metford rifle and another rifle often referred to by what acronymn?
A: {"James Paris Lee is best known for investing the Lee−Metford rifle and another rifle often referred to by what acronymn?": ["James Paris Lee is best known for investing the Lee−Metford rifle and which other rifle?", "<1> is often referred to by what acronymn?"]}.
Q: What year did Edburga of Minster−in−Thanet's father die?
A: {"What year did Edburga of Minster−in−Thanet's father die?": ["Who is Edburga of Minster−in−Thanet's father?", "What year did <1> die?"]}.
Q: Were Lonny and Allure both founded in the 1990s?
A: {"Were Lonny and Allure both founded in the 1990s?": ["When was Lonny (magazine) founded?", "When was Allure founded?"]}.
Q: The actor that stars as Joe Proctor on the series "Power" also played a character on "Entourage" that has what last name?
A: {"The actor that stars as Joe Proctor on the series \"Power\" also played a character on \"Entourage\" that has what last name?": ["Which actor stars as Joe Proctor on the series \"Power\"?", "<1> played a character on \"Entourage\" that has what last name?"]}.
Q: How many awards did the "A Girl Like Me" singer win at the American Music Awards of 2012?
A: {"How many awards did the \"A Girl Like Me\" singer win at the American Music Awards of 2012?": ["Who is the singer of \"A Girl Like Me\"?", "How many awards did <1> win at the American Music Awards of 2012?"]}.
Q: Dadi Denis studied at a Maryland college whose name was changed in 1890 to honor what man?
A: {"Dadi Denis studied at a Maryland college whose name was changed in 1890 to honor what man?": ["Dadi Denis studied at which Maryland college?", "<1>'s name was changed in 1890 to honor what man?"]}.
Q: William Orman Beerman was born in a city in northeastern Kansas that is the county seat of what county?
A: {"William Orman Beerman was born in a city in northeastern Kansas that is the county seat of what county?": ["In which city in northeastern Kansas William Orman Beerman was born?", "<1> is the county seat of what county?"]}.\
"""
    

    tree_generation_prompt = """\
<Instruction>Please generate a hierarchical question decomposition tree (HQDT) with json format for a given question. In this tree, the root node is the original complex question, and each non-root node is a sub-question of its parent. The leaf nodes are atomic questions that cannot be further decomposed.</Instruction>
<Examples>
{examples}
</Examples>
Q: {question}
A: \
"""

    cot_sc_prompt_get_keywords = """\
    <Instruction> Give me a list of keywords for a wikipedia lookup to be able to answer this question. 
    Think Step by Step, and finally give the keywords in the following format: <Keywords>["keyword1", "keyword2"]</Keywords>.
    </Instruction>
    <Question>{input}</Question>
    Output:
    """

    cot_sc_prompt_answer_question = """\
    <Context>{context}</Context>
    <Instruction> Answer the question using the provided context. Think Step by Step, and finally give the answer in this exact format: <Answer>answer</Answer>.
    Do not make the answer more than 5 words.
    <Examples>
    {examples}
    </Examples>
    </Instruction>
    <Question>{question}</Question>
    Output:
    """

    cot_zeroshot_prompt_answer_question = """\
       <Context>{context}</Context>
       <Instruction> Answer the question using the provided context. Think Step by Step, and finally give the answer in this exact format: <Answer>answer</Answer>.
       Do not make the answer more than 5 words.
       </Instruction>
       <Question>{question}</Question>
       Output:
       """

    tot_prompt_get_keywords = """\
    <Instruction> Give me a list of keywords for a wikipedia lookup to be able to answer this question. 
    Think Step by Step, and finally give the keywords in the following format: <Keywords>["keyword1", "keyword2"]</Keywords>.
    </Instruction>
    <Question>{input}</Question>
    Output:
    """

    tot_prompt_verify_retrieved_documents = """\
    <Instruction> To answer the question 
    <Question>{input}</Question>
    You used the list of keywords <Keywords>{keywords}</Keywords> and was able to retrieve the following documents:
    <Documents>
    {documents}
    </Documents>
    
    If the documents retrieved are sufficient to correctly answer the question, give the final answer in this exact format: <Answer>answer</Answer>. 
    Do not make the answer more than 5 words. Otherwise if you need to update the keywords and search again, give the keywords in the following format: <Keywords>["keyword1", "keyword2"]</Keywords>.
    </Instruction>
    """

    tot_prompt_answer_question = """\
        <Context>{context}</Context>
        <Instruction> Answer the question using the provided context. Think Step by Step, and finally give the answer in this exact format: <Answer>answer</Answer>.
        Do not make the answer more than 5 words.
        <Examples>
        {examples}
        </Examples>
        </Instruction>
        <Question>{question}</Question>
        Output:
        """

    plan_solve_answer_question = """\
    <Instruction>Answer the question using the provided context. Only output the final answer directly without any other text.
    {plan_solve_prompt}
    </Instruction>
    <Context>{context}</Context>
    <Question>{question}</Question>
    
    """

    # The Plan and Solve Prompt from Wang et al. (2023)
    plan_solve_basic_prompt = "Let's first understand the problem and devise a plan to solve the problem. " \
                              "Then, let's carry out the plan to solve the problem step by step. " \
                              "Give the final answer in this format: <Answer>answer</Answer>"

    # The Plan and Solve Plus Prompt from Wang et al. (2023)
    plan_solve_plus_prompt = "Let's first understand the problem, extract relevant variables and their corresponding numerals, " \
                             "and make and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables " \
                             "(pay attention to correct numerical calculation and commonsense), " \
                             "solve the problem step by step, and show the answer. " \
                             "Give the final answer in this format: <Answer>answer</Answer>"

    def generate_prompt(self, num_branches: int, original: str, current: str, method: str, **kwargs) -> str:
        assert num_branches == 1, "Branching should be done via multiple requests."
        examples = utils.parse_examples()
        if method.startswith("io"):
            if kwargs["phase"] == 0:
                prompt = self.io_prompt_get_keywords.format(input=original)
            elif kwargs["phase"] == 2:
                examples_str = "\n".join([f"{example["question"]}<Answer>{example["io_answer"]}</Answer>" for example in examples])
                prompt = self.io_prompt_answer_question.format(context=current, question=original, examples=examples_str)
        elif method.startswith("probtree"):
            prompt = self.tree_generation_prompt.format(question=original, examples=self.tree_generation_examples)
        elif method.startswith("cot_sc") or method.startswith("cot"):
            if kwargs["phase"] == 0:
                prompt = self.cot_sc_prompt_get_keywords.format(input=original)
            elif kwargs["phase"] == 2:
                examples_str = "\n".join(
                    [f"{example["question"]}{example["cot_answer"]}<Answer>\n{example["io_answer"]}</Answer>" for example in examples])
                if method == "cot_zeroshot":
                    prompt = self.cot_zeroshot_prompt_answer_question.format(context=current, question=original)
                else:
                    prompt = self.cot_sc_prompt_answer_question.format(context=current, question=original,
                                                                       examples=examples_str)
        elif method == "plan_solve_basic":
            if kwargs["phase"] == 0:
                prompt = self.io_prompt_get_keywords.format(input=original)
            elif kwargs["phase"] == 2:
                prompt = self.plan_solve_answer_question.format(context=current, question=original, plan_solve_prompt = self.plan_solve_basic_prompt)
        elif method == "plan_solve_plus":
            if kwargs["phase"] == 0:
                prompt = self.io_prompt_get_keywords.format(input=original)
            elif kwargs["phase"] == 2:
                prompt = self.plan_solve_answer_question.format(context=current, question=original,
                                                                plan_solve_prompt=self.plan_solve_plus_prompt)
        elif method.startswith("tot"):
            logging.info("phase = {}".format(kwargs["phase"]))
            if kwargs["phase"] == 0:
                prompt = self.tot_prompt_get_keywords.format(input=original)
            elif kwargs["phase"] == 2:
                keywords_str = "["+", ".join([f"\"{keyword}\"" for keyword in kwargs["keywords"]])+"]"
                logging.info("keywords_str: {}".format(keywords_str))
                prompt = self.tot_prompt_verify_retrieved_documents.format(documents=current, input=original, keywords=keywords_str)
            elif kwargs["phase"] == 4:
                examples_str = "\n".join(
                    [f"{example["question"]}{example["cot_answer"]}\n<Answer>{example["io_answer"]}</Answer>" for example
                     in examples])
                prompt = self.tot_prompt_answer_question.format(context=current, question=original, examples = examples_str)

        else:
            raise ValueError(f"generate_prompt: Unknown method: {method}")
        logging.info("full_prompt: %s", prompt)

        return prompt

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def improve_prompt(self, **kwargs) -> str:
        pass

    def validation_prompt(self, **kwargs) -> str:
        pass
    