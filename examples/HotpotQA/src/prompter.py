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
A: {"Vertical Limit stars which actor who also played astronaut Alan Shepard in \\"The Right Stuff\\"?": ["Vertical Limit stars which actor?", "Which actor played astronaut Alan Shepard in \\"The Right Stuff\\"?"]}.
Q: What was the 2014 population of the city where Lake Wales Medical Center is located?
A: {"What was the 2014 population of the city where Lake Wales Medical Center is located?": ["Which city was Lake Wales Medical Center located in?", "What was the 2014 population of <1>?"]}.
Q: Who was born first? Jan de Bont or Raoul Walsh?
A: {"Who was born first? Jan de Bont or Raoul Walsh?": ["When was Jan de Bont born?", "When was Raoul Walsh born?"]}.
Q: In what country was Lost Gravity manufactured?
A: {"In what country was Lost Gravity manufactured?": ["Which company was Lost Gravity manufactured?", "Which country is <1> in?"]}.
Q: Which of the following had a debut album entitled "We Have an Emergency": Hot Hot Heat or The Operation M.D.?
A: {"Which of the following had a debut album entitled \\"We Have an Emergency\\": Hot Hot Heat or The Operation M.D.?": ["What is the debut album of the band Hot Hot Heat?", "What is the debut album of the band The Operation M.D.?"]}.
Q: In which country did this Australian who was detained in Guantanamo Bay detention camp and published "Guantanamo: My Journey" receive para−military training?
A: {"In which country did this Australian who was detained in Guantanamo Bay detention camp and published \\"Guantanamo: My Journey\\" receive para−military training?": ["Which Australian was detained in Guantanamo Bay detention camp and published \\"Guantanamo: My Journey\\"?", "In which country did <1> receive para−military training?"]}.
Q: Does The Border Surrender or Unsane have more members?
A: {"Does The Border Surrender or Unsane have more members?": ["How many members does The Border Surrender have?", "How many members does Unsane have?"]}.
Q: James Paris Lee is best known for investing the Lee−Metford rifle and another rifle often referred to by what acronymn?
A: {"James Paris Lee is best known for investing the Lee−Metford rifle and another rifle often referred to by what acronymn?": ["James Paris Lee is best known for investing the Lee−Metford rifle and which other rifle?", "<1> is often referred to by what acronymn?"]}.
Q: What year did Edburga of Minster−in−Thanet's father die?
A: {"What year did Edburga of Minster−in−Thanet's father die?": ["Who is Edburga of Minster−in−Thanet's father?", "What year did <1> die?"]}.
Q: Were Lonny and Allure both founded in the 1990s?
A: {"Were Lonny and Allure both founded in the 1990s?": ["When was Lonny (magazine) founded?", "When was Allure founded?"]}.
Q: The actor that stars as Joe Proctor on the series "Power" also played a character on "Entourage" that has what last name?
A: {"The actor that stars as Joe Proctor on the series \\"Power\\" also played a character on \\"Entourage\\" that has what last name?": ["Which actor stars as Joe Proctor on the series \\"Power\\"?", "<1> played a character on \\"Entourage\\" that has what last name?"]}.
Q: How many awards did the "A Girl Like Me" singer win at the American Music Awards of 2012?
A: {"How many awards did the \\"A Girl Like Me\\" singer win at the American Music Awards of 2012?": ["Who is the singer of \\"A Girl Like Me\\"?", "How many awards did <1> win at the American Music Awards of 2012?"]}.
Q: Dadi Denis studied at a Maryland college whose name was changed in 1890 to honor what man?
A: {"Dadi Denis studied at a Maryland college whose name was changed in 1890 to honor what man?": ["Dadi Denis studied at which Maryland college?", "<1>'s name was changed in 1890 to honor what man?"]}.
Q: William Orman Beerman was born in a city in northeastern Kansas that is the county seat of what county?
A: {"William Orman Beerman was born in a city in northeastern Kansas that is the county seat of what county?": ["In which city in northeastern Kansas William Orman Beerman was born?", "<1> is the county seat of what county?"]}.\
"""

#     tree_generation_examples_long_tree = """\
# Q: When did the director of film Hypocrite (Film) die?
# A: {"When did the director of film Hypocrite (Film) die?": ["Who is the director of film Hypocrite (Film)?", "When did #1 die?"]}.
# Q: Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?
# A: {"Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?": ["What is the nationality of the director of film Coolie No. 1 (1995 Film)?", "What is the nationality of the director of film The Sensational Trial?"], "What is the nationality of the director of film Coolie No. 1 (1995 Film)?": ["Who is the director of film Coolie No. 1 (1995 Film)?", "What is the nationality of #1?"], "What is the nationality of the director of film The Sensational Trial?": ["Who is the director of film The Sensational Trial?", "What is the nationality of #1?"]}.
# Q: Are both Kurram Garhi and Trojkrsti located in the same country?
# A: {"Are both Kurram Garhi and Trojkrsti located in the same country?": ["Which country is Kurram Garhi located in?", "Which country is Trojkrsti located in?"]}.
# Q: Who was born first out of Martin Hodge and Ivania Martinich?
# A: {"Who was born first out of Martin Hodge and Ivania Martinich?": ["When was Martin Hodge born?", "When was Ivania Martinich born?"]}.
# Q: Which film came out first, The Night Of Tricks or The Genealogy?
# A: {"Which film came out first, The Night Of Tricks or The Genealogy?": ["When was the film The Night Of Tricks published?", "When was the film The Genealogy published?"]}.
# Q: When did the director of film Laughter In Hell die?
# A: {"When did the director of film Laughter In Hell die?": ["Who is the director of film Laughter In Hell?", "When did #1 die?"]}.
# Q: Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?
# A: {"Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?": ["When did the director of film The Gal Who Took the West die?", "When did the director of film Twenty Plus Two die?"], "When did the director of film The Gal Who Took the West die?": ["Who is the director of film The Gal Who Took the West?", "When did #1 die?"], "When did the director of film Twenty Plus Two die?": ["Who is the director of film Twenty Plus Two?", "When did #1 die?"]}.
# Q: Who is Boraqchin (Wife Of ÃUgedei)'s father−in−law?
# A: {"Who is Boraqchin (Wife Of ÃUgedei)'s father−in−law?": ["Who is Boraqchin married to?", "Who is the father of #1?"]}.
# Q: What is the cause of death of Grand Duke Alexei Alexandrovich Of Russia's mother?
# A: {"What is the cause of death of Grand Duke Alexei Alexandrovich Of Russia's mother?": ["Who is the mother of Grand Duke Alexei Alexandrovich Of Russia?", "What is the cause of death of #1?"]}.
# Q: Which film has the director died earlier, When The Mad Aunts Arrive or The Miracle Worker (1962 Film)?
# A: {"Which film has the director died earlier, When The Mad Aunts Arrive or The Miracle Worker (1962 Film)?": ["When did the director of film When The Mad Aunts Arrive die?", "When did the director of film The Miracle Worker (1962 Film) die?"], "When did the director of film When The Mad Aunts Arrive die?": ["Who is the director of film When The Mad Aunts Arrive?", "When did #1 die?"], "When did the director of film The Miracle Worker (1962 Film) die?": ["Who is the director of film The Miracle Worker (1962 Film)?", "When did #1 die?"]}.
# Q: Which album was released earlier, What'S Inside or Cassandra'S Dream (Album)?
# A: {"Which album was released earlier, What'S Inside or Cassandra'S Dream (Album)?": ["When was the album What'S Inside released?", "When was the album Cassandra'S Dream (Album) released?"]}.
# Q: Are both mountains, Serre Mourene and Monte Galbiga, located in the same country?
# A: {"Are both mountains, Serre Mourene and Monte Galbiga, located in the same country?": ["Which country was the mountain Serre Mourene located in?", "Which country was the mountain Monte Galbiga located in?"]}.
# Q: What is the date of birth of the director of film Best Friends (1982 Film)?
# A: {"What is the date of birth of the director of film Best Friends (1982 Film)?": ["Who is the director of film Best Friends (1982 Film)?", "What is the date of birth of #1?"]}.
# Q: Which film has the director born first, Two Weeks With Pay or Chhailla Babu?
# A: {"Which film has the director born first, Two Weeks With Pay or Chhailla Babu?": ["When was the director of film Two Weeks With Pay born?", "When was the director of film Chhailla Babu born?"], "When was the director of film Two Weeks With Pay born?": ["Who is the director of film Two Weeks With Pay?", "When was #1 born?"], "When was the director of film Chhailla Babu born?": ["Who is the director of film Chhailla Babu?", "When was #1 born?"]}.
# Q: Who is the grandchild of Krishna Shah (Nepalese Royal)?
# A: {"Who is the grandchild of Krishna Shah (Nepalese Royal)?": ["Who is the child of Krishna Shah (Nepalese Royal)?", "Who is the child of #1?"]}.
# Q: When was the director of film P.S. Jerusalem born?
# A: {"When was the director of film P.S. Jerusalem born?": ["Who is the director of film P.S. Jerusalem?", "When was #1 born?"]}.
# Q: Which album was released more recently, If I Have to Stand Alone or Answering Machine Music?
# A: {"Which album was released more recently, If I Have to Stand Alone or Answering Machine Music?": ["When was the album If I Have to Stand Alone released?", "When was the album Answering Machine Music released?"]}.
# Q: Where did the director of film Maddalena (1954 Film) die?
# A: {"Where did the director of film Maddalena (1954 Film) die?": ["Who is the director of film Maddalena (1954 Film)?", "Where did #1 die?"]}.
# Q: When did the director of film The Boy And The Fog die?
# A: {"When did the director of film The Boy And The Fog die?": ["Who is the director of film The Boy And The Fog?", "When did #1 die?"]}.
# Q: Are the directors of films The Sun of the Sleepless and Nevada (1927 film) both from the same country?
# A: {"Are the directors of films The Sun of the Sleepless and Nevada (1927 film) both from the same country?": ["Which country is the director of film The Sun of the Sleepless from?", "Which country is the director of film Nevada (1927 film) from?"], "Which country is the director of film The Sun of the Sleepless from?": ["Who is the director of film The Sun of the Sleepless?", "Which country is #1 from?"], "Which country is the director of film Nevada (1927 film) from?": ["Who is the director of film Nevada (1927 film)?", "Which country is #1 from?"]}.\
# """

    tree_generation_prompt = """\
Please generate a hierarchical question decomposition tree (HQDT) with json format for a given question. In this tree, the root node is the original complex question, and each non-root node is a sub-question of its parent. The leaf nodes are atomic questions that cannot be further decomposed.
{examples}
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
            if kwargs.get("openbook_qa", False):
                prompt = self.open_book_generation_prompt.format(question=kwargs["question"], examples=self.open_book_examples, context=kwargs["context"])
            elif kwargs.get("closedbook_qa", False):
                prompt = self.closed_book_generation_prompt.format(question=kwargs["question"], examples=self.closed_book_examples)
            elif kwargs.get("child_aggregating", False):
                prompt = self.child_aggregating_prompt.format(question=kwargs["question"], context=kwargs["context"], examples=self.child_aggregating_examples)
            else:
                prompt = self.tree_generation_prompt.format(question=original, examples=f"{self.tree_generation_examples}\n{self.tree_generation_examples}")
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
    