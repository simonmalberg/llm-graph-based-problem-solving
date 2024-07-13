import json
from termcolor import colored
from openai.types.chat.chat_completion import ChatCompletion
from typing import List, Any

# def build_dict_from_str_list(input_list: List[str, Any]) -> List[Any]:
#     # Concatenate all strings to form the JSON string
#     json_string = "".join([item[0] for item in input_list])
    
#     # Parse the JSON string to form a dictionary
#     try:
#         parsed_dict = json.loads(json_string)
#     except json.JSONDecodeError as e:
#         print(f"Error decoding JSON: {e}")
#         return []

#     # Convert the dictionary back to a list of [str, Any] pairs
#     output_list = [[key, value] for key, value in parsed_dict.items()]
    
#     return output_list


def parse_tree_and_extract_logprobs(llm_response: ChatCompletion):
    # prompt = llm_response['prompt']
    # question = prompt.split('\n')[-2][len('Q: '):].strip()
    # print(colored(question, 'red'))
    # print(item['response']['text'])
    try:
        qds = llm_response.choices[0].message.content.strip()
        if qds.endswith('.'):
            qds = qds[:-1]
        hqdt = json.loads(qds)
    except:
        hqdt = None
    



    tokens = [per_token.token for per_token in llm_response.choices[0].logprobs.content]
    token_logprobs = [per_token.logprob for per_token in llm_response.choices[0].logprobs.content]
    if len(token_logprobs) == 0:
        return

    if tokens[-1] == '.':
        token_logprobs = token_logprobs[:-1]
        # print(answer_logprobs)
    # else:
    #     answer_logprobs = token_logprobs[pos+6:]

    # print(tokens[pos+6:-1])
    
    st, ed = 0, 0
    pos = 0
    qds = {}
    for sub_question, qd in hqdt.items():
        while pos < len(tokens):
            #print("".join(tokens[max(pos-1, 0): min(pos+2, len(tokens))]))
            # if "[" in tokens[pos] and ": [\"" in "".join(tokens[max(pos-1, 0): min(pos+2, len(tokens))]):
            if "[" in tokens[pos]:
                st = pos
                break
            pos += 1
        while pos < len(tokens):
            # if "]" in tokens[pos] and "\"]" in "".join(tokens[max(pos-1, 0): min(pos+2, len(tokens))]):
            if "]" in tokens[pos]:
                ed = pos
                break
            pos += 1
        assert pos < len(tokens), str(st) + " | " + str(ed)
        qd_score = sum(token_logprobs[st:ed+1]) / len(token_logprobs[st:ed+1])
        if any([x == sub_question for x in qd]):
            qd, qd_score = [], None
        qds[sub_question] = (qd, qd_score)
        print(colored(sub_question, 'blue'))
        print("".join(tokens[st:ed+1]))
    
    
    # answer_logprob = sum(token_logprobs) / len(token_logprobs)
    # data[question] = [hqdt, answer_logprob]
    return qds
