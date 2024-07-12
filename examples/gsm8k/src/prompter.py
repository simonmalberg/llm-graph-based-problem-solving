import logging
from typing import Dict, List
from graph_of_thoughts import prompter


class GSM8KPrompter(prompter.Prompter):
    """
    GSM8KPrompter provides the generation of prompts specific to the gsm8k example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    io_prompt_base = """\
    <Instruction> Solve the following math problems and provide ONLY the integer solution with no comma or dot. Do not output any thoughts, only the answer after. </Instruction>

    <Examples>
    Input: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
    Output: 72

    Input: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
    Output: 10

    Input: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
    Output: 5

    Input: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
    Output: 42

    Input: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
    Output: 624
    </Examples>

    Input: {input}
    Output: """

    cot_prompt_base = """\
    <Instruction> Solve the following math problems and provide the full reasoning in the answer as well as the integer solution behind ####  with no comma or dot. </Instruction>

    <Examples>
    Input: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
    Output: Natalia sold 48/2 = 24 clips in May.
    Natalia sold 48+24 = 72 clips altogether in April and May.
    #### 72

    Input: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
    Output: Weng earns 12/60 = $0.2 per minute.
    Working 50 minutes, she earned 0.2 x 50 = $10.
    #### 10

    Input: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
    Output: In the beginning, Betty has only 100 / 2 = $50.
    Betty's grandparents gave her 15 * 2 = $30.
    This means, Betty needs 100 - 50 - 30 - 15 = $5 more.
    #### 5

    Input: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
    Output: Maila read 12 x 2 = 24 pages today.
    So she was able to read a total of 12 + 24 = 36 pages since yesterday.
    There are 120 - 36 = 84 pages left to be read.
    Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = 42 pages.
    #### 42

    Input: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
    Output: He writes each friend 3*2=6 pages a week.
    So he writes 6*2=12 pages every week.
    That means he writes 12*52=624 pages a year.
    #### 624
    </Examples>

    Input: {input}
    Output: """

    plan_and_solve_prompt_base = """\
    <Instruction> Plan and solve the following math problems and provide the full reasoning in the answer as well as the integer solution behind ####  with no comma or dot. </Instruction>

    <Examples>
    Input: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
    Plan:
        Step 1: How many clips did Natalia sell in April?
        Step 2: How many clips did Natalia sell in May?
        Step 3: How many clips did Natalia sell altogether in April and May?
    Solution:
        Step 1: Natalia sold 48 clips in April.
        Step 2: Natalia sold 48/2 = 24 clips in May.
        Step 3: Natalia sold 48+24 = 72 clips altogether in April and May.
    #### 72

    Input: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
    Plan:
        Step 1: How much does Weng earn per minute?
        Step 2: How much did Weng earn in 50 minutes?
    Solution:
        Step 1: Weng earns 12/60 = $0.2 per minute.
        Step 2: Working 50 minutes, she earned 0.2 x 50 = $10.
    #### 10

    Input: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
    Plan:
        Step 1: How much money does Betty have in the beginning?
        Step 2: How much money did Betty's grandparents give her?
        Step 3: How much money does Betty have in total now?
        Step 4: How much more money does Betty need to buy the wallet?
    Solution:
        Step 1: In the beginning, Betty has only 100 / 2 = $50.
        Step 2: Betty's grandparents gave her 15 * 2 = $30.
        Step 3: This means, Betty needs 100 - 50 - 30 - 15 = $5 more.
    #### 5

    Input: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
    Plan:
        Step 1: How many pages did Julie read today?
        Step 2: How many pages did Julie read since yesterday?
        Step 3: How many pages are left to be read?
        Step 4: How many pages should Julie read tomorrow?
    Solution:
        Step 1: Julie read 12 x 2 = 24 pages today.
        Step 2: So she was able to read a total of 12 + 24 = 36 pages since yesterday.
        Step 3: There are 120 - 36 = 84 pages left to be read.
        Step 4: Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = 42 pages.
    #### 42

    Input: James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?
    Plan:
        Step 1: How many pages does he write to each friend a week?
        Step 2: How many pages does he write to each friend a year?
        Step 3: How many pages does he write to both friends a year?
    Solution:
        Step 1: He writes each friend 3*2=6 pages a week.
        Step 2: There are 52 weeks in a year, so he writes 6*52=312 pages to each friend a year.
        Step 3: That means he writes 312*2=624 pages a year.
    #### 624
    </Examples>

    Input: {input}
    Plan:
    """

    score_prompt_base = """\
    <Instruction> Score the answer given for the following (partial) math problem. The answer should be binary, True if the answer is correct and False if it is incorrect. Output only True or False. </Instruction>
    Input (partial) math problem: {input}
    Answer to the math problem: {answer}
    Output: """


    tot_initial_prompt = """
    <Instruction> For the following math problems, list the variables mentioned in the question, and then formulate the \
    equations required to arrive at the solution.
    <Example>
    Input: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
    Output:
    total_pages = 120
    pages_yesterday = 12
    pages_today = 2 * pages_yesterday
    remaining_pages = total_pages - (pages_yesterday + pages_today) = total_pages - (3 * pages_yesterday)
    answer = pages_tomorrow = remaining_pages / 2 = (total_pages - (3 * pages_yesterday)) / 2
    </Example>
    </Instruction>
    
    Input: {input}
    Output: """

    tot_initial_vote = """
    <Instruction> For the given question, the variables and equations required to arrive at the solution are listed. Score the formulation between 1 - 10
    based on the confidence you have on the correctness of the values and the steps.
    <Example>
    Input: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
    Setup: 
    total_pages = 120
    pages_yesterday = 12
    pages_today = 2 * pages_yesterday
    remaining_pages = total_pages - (pages_yesterday + pages_today) = total_pages - (3 * pages_yesterday)
    answer = pages_tomorrow = remaining_pages / 2 = (total_pages - (3 * pages_yesterday)) / 2
    Output:
    <Score>9</Score>
    </Example>
    </Instruction>
    Input: {input}
    Setup: {setup}
    Output: """

    tot_solve_prompt = """
    <Instruction> Given the following variables and formulae, solve for the answer and give the integer solution behind ####  with no comma or dot.
    <Example>
    Input: total_pages = 120
    pages_yesterday = 12
    pages_today = 2 * pages_yesterday
    remaining_pages = total_pages - (pages_yesterday + pages_today) = total_pages - (3 * pages_yesterday)
    answer = pages_tomorrow = remaining_pages / 2 = (total_pages - (3 * pages_yesterday)) / 2
    
    Output:
    answer = (total_pages - (3 * pages_yesterday)) / 2 = (120 - (3*12))/2 = 42
    #### 42
    </Example>
    </Instruction>
    Input: {input}
    Output: """

    tot_final_vote = """
    <Instruction> Given the following problem and solution, evaluate whether the solution is correct or not with True or False.
    <Example>
    Input: Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?
    Solution: answer = (total_pages - (3 * pages_yesterday)) / 2 = (120 - (3*12))/2 = 42
    Output: True
    </Example>
    
    Input: {input}
    Solution: {solution}
    Output: """

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

        full_prompt = ""
        if method.startswith("io"):
            full_prompt =  self.io_prompt_base.format(input=input)
        elif method.startswith("cot"):
            full_prompt =  self.cot_prompt_base.format(input=input)
        elif method.startswith("plan_and_solve"):
            full_prompt =  self.plan_and_solve_prompt_base.format(input=input)
        elif method == "tot":
            if current is None or current == "":
                full_prompt = self.tot_initial_prompt.format(input=input)
            else:
                full_prompt = self.tot_solve_prompt.format(input=input)

        else:
            raise ValueError(f"Unknown method: {method}")

        return full_prompt

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        method = state_dicts[0]["method"]
        phase = state_dicts[0]["phase"]
        logging.info("score_prompt: phase = {}".format(phase))
        full_prompt = ""
        if method == "tot":
            if phase < 3:
                full_prompt = self.tot_initial_vote.format(input=state_dicts[0]["original"], setup=state_dicts[0]["current"])
            else:
                full_prompt = self.tot_final_vote.format(input=state_dicts[0]["original"], solution=state_dicts[0]["current"])
        else:
            raise ValueError(f"Scoring not implemented for method: {method}")
        logging.info("full_prompt: {}".format(full_prompt))
        return full_prompt

    def improve_prompt(self, **kwargs) -> str:
        pass

    def validation_prompt(self, **kwargs) -> str:
        pass
    