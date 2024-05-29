from typing import Dict


def error(state: Dict) -> float:
    """
    Function to locally count the number of errors that serves as a score.

    :param state: Thought state to be scored.
    :type state: Dict
    :return: Number of errors.
    :rtype: float
    """

    try:
        unsorted_list = state["original"]
        if (
            "unsorted_sublist" in state
            and state["unsorted_sublist"] != ""
            and state["unsorted_sublist"] is not None
            and len(state["unsorted_sublist"]) < len(unsorted_list) - 5
        ):
            unsorted_list = state["unsorted_sublist"]
        correct_list = sorted(string_to_list(unsorted_list))
        current_list = string_to_list(state["current"])
        num_errors = 0
        for i in range(10):
            num_errors += abs(
                sum([1 for num in current_list if num == i])
                - sum([1 for num in correct_list if num == i])
            )
        num_errors += sum(
            [1 for num1, num2 in zip(current_list, current_list[1:]) if num1 > num2]
        )
        return num_errors
    except:
        return 300