import logging
from typing import Dict, List

from graph_of_thoughts import prompter
from graph_of_thoughts.operations import Thought


class CommonsenseQAPrompter(prompter.Prompter):
    """
    CommonsenseQAPrompter provides the generation of prompts specific to the commonsenseQA example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    io_zeroshot_prompt = """<Instruction> Use your commonsense knowledge to answer the following question. Choose the correct answer from the options provided below. Output the answer in the following format: 
        <Answer>answer</Answer>
        </Instruction>

        Input: {input}
        Output:"""


    io_prompt = """<Instruction> Use your commonsense knowledge to answer the following question. Choose the correct answer from the options provided below. Output the answer in the following format: 
    <Answer>answer</Answer>
    </Instruction>

    <Examples>
    Input: 
    Question: What item is most useful for keeping your pants from falling down?
    A) shirt
    B) belt
    C) watch
    D) shoes
    E) hat
    Output: <Answer>B</Answer>
    </Examples>

    Input: {input}
    Output:"""

    cot_prompt = """<Instruction> Use your commonsense knowledge to answer the following question. Let us think Step by Step, then provide the answer in this format: <Answer>answer</Answer>
    </Instruction>
    
    <Approach>
    To give the best answer follow these steps:
    1.clearly state the letter of the answer in the format given.
    2.You need to first paraphrase the problem, state the relevant premise according to the context, 
    3.Deduct facts one at a step.Give the reason why the answer is correct.
    </Approach>

    <Examples>
    Input: 
    Question: Where is a business restaurant likely to be located??
    A) town
    B) at hotel
    C) mall
    D) business sector
    E) yellow pages
    Output:
    Business restaurants are designed to cater to individuals who are involved in business activities, often looking for convenience and efficiency during business hours.
    Town (A): Could include various zones, not necessarily linked to business activities.
    At Hotel (B): Could serve business travelers but is not exclusively for business professionals and might cater more to tourists.
    Mall (C): Focuses more on retail and family-oriented services rather than business dealings.
    Business Sector (D): Directly caters to the business crowd, located within or near business hubs and offices.
    Yellow Pages (E): Not a physical location but a business directory.
    Therefore the answer is 
    <Answer>D</Answer>

    Input: 
    Question: When someone doesn't know how to skate well, they normally do what to stay up?
    A) spin
    B) romance
    C) hold hands
    D) fall down
    E) grab side railing
    Output:
    Individuals who are not proficient in skating often need support to maintain balance and prevent falls. This support can come in various forms, depending on what is available and what the individual feels most comfortable with.
    Spin (A): This is a complex move usually performed by experienced skaters, not typical for beginners.
    Romance (B): This choice is unrelated to physical support for staying upright while skating.
    Hold hands (C): This is a common method for beginners to support each other and maintain balance.
    Fall down (D): This is a result of losing balance, not a method to stay up.
    Grab side railing (E): This provides physical support and is a common choice for beginners to help themselves stay upright.
    Therefore the answer is
    <Answer>E</Answer>
    </Examples>

    Input: {input}
    Output: """

    cot_zeroshot_prompt = """<Instruction> Use your commonsense knowledge to answer the following question. Let us think Step by Step, then provide the answer in this format: 
    <Answer>answer</Answer>
    e.g. <Answer>E</Answer>
    </Instruction>

    Input: {input}
    Output:"""

    plan_solve_basic_prompt = """<Instruction> Use your commonsense knowledge to answer the following question. 
    Let's first understand the problem and devise a plan to solve the problem. 
    Then, let's carry out the plan to solve the problem step by step.
    Give the final answer in this format: <Answer>answer</Answer>
    e.g. <Answer>E</Answer>
    </Instruction>
    Input: {input}
    Output:"""

    # The Plan and Solve Plus Prompt from Wang et al. (2023)
    plan_solve_plus_prompt = """<Instruction> Use your commonsense knowledge to answer the following question. 
    Let's first understand the problem, extract relevant variables and their corresponding numerals, and make and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables (pay attention to correct numerical calculation and commonsense), 
    solve the problem step by step, and show the answer.
    Give the final answer in this format: <Answer>answer</Answer>
    e.g. <Answer>E</Answer>
    </Instruction>  
    Input: {input}
    Output:"""

    tot_base_prompt = """<Instruction>
    The following two inputs represent an initial input question and preliminary explanation of the problem 
    with a rough choice answer. The answer might be incorrect and parsing may be misconstrued or too narrowly focused 
    Fix the answer such that it has the correct parsing and most likely to be the correct option. Output answers in format given:
    <Answer>answer</Answer>
    </Instruction>

    <Approach>
    To give the best answer follow these steps:
    1.Test the answer.
    2.Iterate through the input text, paraphrase it and identify the key points questions.
    3.State the relevant premise according to the context.Deduct facts one at a step. Give the reason why the answer is incorrect (or correct).
    4.Compare with current answers and update if the current answer is wrong
    </Approach>

    <Examples>
    Input: 
    Question:In what Spanish speaking North American country can you get a great cup of coffee?
    A) mildred's coffee shop
    B) mexico
    C) diner
    D) kitchen
    E) canteen
    Current Answer: 
    The question is asking where you can find a great cup of coffee in a Spanish-speaking North American country.
    Deduct facts:
    - Mildred's Coffee Shop (A):it is a likely place to find a great cup of coffee instead of a country.
    - Mexico (B): Mexico is a Spanish-speaking country in North America, and famous for its coffee.
    - Diner (C): Diners may serve coffee, but they are not specifically known for providing great coffee.
    - Kitchen (D): Kitchens are not typically places where you can get a cup of coffee.
    - Canteen (E): Canteens may offer coffee, but they are not usually associated with providing great coffee like a coffee shop would.
    Therefore the answer is <Answer>B</Answer>
    Output: 
    <Answer>B</Answer>
    Reason: The answer is right. The question is asking for a answer about which Spanish-speaking North American country instead of a place.

        
    Input: 
    Question: Where is a business restaurant likely to be located??
    A) town
    B) at hotel
    C) mall
    D) business sector
    E) yellow pages
    Current Answer: 
    Business restaurants are designed to cater to individuals who are involved in business activities, often looking for convenience and efficiency during business hours.
    Town (A): Could include various zones, not necessarily linked to business activities.
    At Hotel (B): Could serve business travelers but is not exclusively for business professionals and might cater more to tourists.
    Mall (C): Focuses more on retail and family-oriented services rather than business dealings.
    Business Sector (D): Directly caters to the business crowd, located within or near business hubs and offices.
    Yellow Pages (E): Not a physical location but a business directory.
    Therefore the answer is 
    <Answer>D</Answer>
    Output:
    <Answer>D</Answer>
    Reason: The answer D is correct because it directly addresses the question, identifying the business sector as the typical location for business restaurants, which cater specifically to business professionals.
    
    </Examples>

    Input: 
    {input}
    Current answer: 
    {current_answer}
    output:
    """

    tot_style_prompt = """<Instruction>
    Imagine three different experts are answering this question.All experts will write down 1 step of their thinking,then share it with the group.Then all experts will go on to the next step, etc.
    If any expert realizes they're wrong at any point then they leave. Output the answer in this format:<Answer>answer</Answer> Anaylyzing:....
    </Instruction>

    <Examples>
    Input:
    Question: In what Spanish speaking North American country can you get a great cup of coffee?
    A) mildred's coffee shop
    B) mexico
    C) diner
    D) kitchen
    E) canteen
    Output:
    <Answer>B</Answer>
    Analyzing:
    Expert 1: Let's consider the Spanish-speaking countries in North America.
    Expert 2: Yes, we need to identify which of those countries are known for their coffee.
    Expert 3: The answer choices include some locations that are not countries, so we can eliminate A, C, D, and E.
    Expert 1: The only Spanish-speaking country in North America listed is Mexico.
    Expert 2: Mexico is known for producing great coffee.
    Expert 3: Therefore, the correct answer must be Mexico.
    All three experts agree that the answer is Mexico.
    </Examples>

    Input: {input}
    Output: """

    score_prompt_base = """
    Given the question, score the given current answers from 1 to 5 based on common sense. Give the score in the format <Score>score</Score>. 
    Use the following criteria:
    - 5: The answer is completely correct and makes perfect common sense.
    - 4: The answer is mostly correct but might have minor inaccuracies.
    - 3: The answer is partially correct but has some noticeable issues.
    - 2: The answer is mostly incorrect but has a small element of truth.
    - 1: The answer is completely incorrect and does not make sense.

    </Instruction>
    <Examples>
    Input: 
    Question: What item is most useful for keeping your pants from falling down?
    A) shirt
    B) belt
    C) watch
    D) shoes
    E) hat
    Current answer:answerKey:"B", because a belt is specifically designed to hold up pants, making it the most useful item for preventing pants from falling down.
    output:<Score>5</Score>

    Input: 
    Question: What item is most useful for telling time?
    A) clock
    B) spoon
    C) knife
    D) scissors
    E) tape
    Current answer:answerKey:"E", because tape is useful for telling time.
    output:<Score>1</Score>

    Input: 
    Question: What item is most useful for writing on a blackboard?
    A) chalk
    B) pen
    C) paper
    D) eraser
    E) marker
    Current answer:answerKey:"B", because pen can write on a blackboard.
    output:<Score>2</Score>

    </Examples>

    Input: 
    {input}
    Current answer: 
    {current_answer}
    output:
    """

    def generate_prompt(self, num_branches: int, original: str, current: str, method: str, **kwargs) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param original: Input text.
        :type original: str
        :param current: Intermediate solution.
        :type current: str
        :param method: Method for which the generate prompt is generated.
        :type method: str
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        :raise AssertionError: If the requested number of branches is not one.
        """
        assert num_branches == 1, "Branching should be done via multiple requests."
        if current is None or current == "":
            input = original
        else:
            input = current

        logging.info(f"Generating prompt with method: {method}, current: {current}")
        if method=="io":
            return self.io_prompt.format(input=input)
        elif method == "io_zeroshot":
            return self.io_zeroshot_prompt.format(input=input)
        elif method == "cot" or method == "cot_sc":
            return self.cot_prompt.format(input=input)
        elif method == "cot_zeroshot":
            return self.cot_zeroshot_prompt.format(input=input)
        elif method == "plan_solve":
            return self.plan_solve_basic_prompt.format(input=input)
        elif method == "plan_solve_plus":
            return self.plan_solve_plus_prompt.format(input=input)
        elif method == "tot_style":
            return self.tot_style_prompt.format(input=input)
        elif method == "tot_base":
            if current is None or current == "":
                return self.cot_prompt.format(input=input)
            return self.tot_base_prompt.format(
                input=original,
                current_answer=current
            )

        else:
            raise ValueError(f"Unknown method: {method}")

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def improve_prompt(self, **kwargs) -> str:
        pass

    def validation_prompt(self, **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        thoughts: List[Thought] = [state_dict["current"] for state_dict in state_dicts]
        inputs = [state_dict["original"] for state_dict in state_dicts]
        final_prompt = self.score_prompt_base.format(input=inputs[0], current_answer=thoughts[0])
        return final_prompt
