from typing import Dict, List
from graph_of_thoughts import prompter


class GSM8KPrompter(prompter.Prompter):
    """
    GSM8KPrompter provides the generation of prompts specific to the gsm8k example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    io_prompt_zeroshot_base = """\
    Provide the answer in the exact format as given.
    <Instruction>Solve the following math problems and provide ONLY the integer solution with no comma or dot. Do not output any thoughts, only the answer after.</Instruction>
    <Input>{input}</Input>
    Output: """

    io_prompt_base = """\
    Provide the answer in the exact format as given.
    <Instruction>Solve the following math problems and provide ONLY the integer solution with no comma or dot. Do not output any thoughts, only the answer after.</Instruction>

    <Examples>
    <Input>Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?</Input>
    Output: <Answer>72</Answer>

    <Input>Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?</Input>
    Output: <Answer>10</Answer>

    <Input>Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?</Input>
    Output: <Answer>5</Answer>

    <Input>Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?</Input>
    Output: <Answer>42</Answer>

    <Input>James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?</Input>
    Output: <Answer>624</Answer>
    </Examples>

    <Input>{input}</Input>
    Output: """

    cot_prompt_base = """\
    Provide the answer in the exact format as given.
    <Instruction> Solve the following math problems and provide the full reasoning as well as the final integer answer without comma or dot in the following format: <Answer>answer</Answer>. </Instruction>

    <Examples>
    <Input>Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?</Input>
    Output: Natalia sold 48/2 = 24 clips in May.
    Natalia sold 48+24 = 72 clips altogether in April and May.
    <Answer>72</Answer>

    <Input>Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?</Input>
    Output: Weng earns 12/60 = $0.2 per minute.
    Working 50 minutes, she earned 0.2 x 50 = $10.
    <Answer>10</Answer>

    <Input>Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?</Input>
    Output: In the beginning, Betty has only 100 / 2 = $50.
    Betty's grandparents gave her 15 * 2 = $30.
    This means, Betty needs 100 - 50 - 30 - 15 = $5 more.
    <Answer>5</Answer>

    <Input>Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?</Input>
    Output: Maila read 12 x 2 = 24 pages today.
    So she was able to read a total of 12 + 24 = 36 pages since yesterday.
    There are 120 - 36 = 84 pages left to be read.
    Since she wants to read half of the remaining pages tomorrow, then she should read 84/2 = 42 pages.
    <Answer>42</Answer>

    <Input>James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?</Input>
    Output: He writes each friend 3*2=6 pages a week.
    So he writes 6*2=12 pages every week.
    That means he writes 12*52=624 pages a year.
    <Answer>624</Answer>
    </Examples>

    <Input>{input}</Input>
    Output: """

    plan_solve_basic_prompt = "Let's first understand the problem and devise a plan to solve the problem. " \
                              "Then, let's carry out the plan to solve the problem step by step. " \
                              "Give the final integer answer with no dot or comma in this format: <Answer>answer</Answer>"

    # The Plan and Solve Plus Prompt from Wang et al. (2023)
    plan_solve_plus_prompt = "Let's first understand the problem, extract relevant variables and their corresponding numerals, " \
                             "and make and devise a complete plan. Then, let's carry out the plan, calculate intermediate variables " \
                             "(pay attention to correct numerical calculation and commonsense), " \
                             "solve the problem step by step, and show the answer. " \
                             "Give the final integer answer with no dot or comma in this format: <Answer>answer</Answer>"


    plan_and_solve_prompt_base = """\
    {plan_and_solve_prompt}
    <Instruction> Solve the following math problem:</Instruction>
    <Input>{input}</Input>
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
        
        if method.startswith("io"):
            return self.io_prompt_base.format(input=input)
        elif method.startswith("cot"):
            return self.cot_prompt_base.format(input=input)
        elif method.startswith("plan_and_solve"):
            if method.endswith("basic"):
                return self.plan_and_solve_prompt_base.format(input=input, plan_and_solve_prompt=self.plan_solve_basic_prompt)
            elif method.endswith("plus"):
                return self.plan_and_solve_prompt_base.format(input=input, plan_and_solve_prompt=self.plan_solve_plus_prompt)
            else:
                raise ValueError(f"Unknown method: {method}")
        else:
            raise ValueError(f"Unknown method: {method}")
        

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        pass

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        raise NotImplementedError("Scoring is not implemented for GSM8K.")

    def improve_prompt(self, **kwargs) -> str:
        pass

    def validation_prompt(self, **kwargs) -> str:
        pass
    