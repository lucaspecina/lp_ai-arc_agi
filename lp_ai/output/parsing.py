import ast
import numpy as np
from langchain_core.messages import AIMessage
from typing import List
import json
import random

def check_output(tool_output):
    # print("\n----------------------------------------------\nTool Output:\n")
    # print(tool_output)
    
    # TODO: remove after debug
    # random_error = random.randint(0, 1)
    # tool_output["parsed"] = None if random_error == 0 else tool_output["parsed"]
    # print(f"random_error: {random_error}")

    if tool_output["parsing_error"]:
        print("Parsing error!")
        raw_output = tool_output
        error = tool_output["parsing_error"]
        raise ValueError(
            f"Error parsing your output! Be sure to invoke the tool. Output: {raw_output}. \n Parse error: {error}"
        )
    elif not tool_output["parsed"]:
        print("Failed to invoke tool!")
        raise ValueError(
            "You did not use the provided tool! Be sure to invoke the tool to structure the output."
        )
    return tool_output

def parse_output(solution):
    # print("Final Parsed Output:", solution["parsed"])
    return solution["parsed"]

def insert_errors(inputs):
    """Insert errors for tool parsing in the messages"""
    # Get errors
    print('Inserting errors -> FALLBACK CHAIN')
    error = inputs["error"]
    messages = inputs["messages"]
    messages += [
        (
            "user",
            f"Retry. You are required to fix the parsing errors: {error} \n\n You must invoke the provided tool.",
        )
    ]
    return {
        "llm_name": inputs["llm_name"],
        "messages": messages,
    }

# # Custom parser for list of strings (patterns)
# def extract_list(message: dict) -> List[str]:
#     """
#     Extract a list of strings from a message content.
#     """
#     try:
#         patterns_str = message['raw'].tool_calls[0]['args']['patterns']
#         patterns_list = json.loads(patterns_str)
#         # message['raw'].tool_calls[0]['args']['patterns'] = patterns_list
#         # set parsing error to None and parsed to True
#         message['parsing_error'] = None
#         message["parsed"] = patterns_list
#         return message
#     except:
#         raise ValueError(f"Failed to extract list of patterns: {message}")

def validate_output(test_task, solutions, task_id, test_output):
    prediction = ast.literal_eval(test_output)
    print('input:', test_task['input'])
    print('output:', np.array(solutions[task_id][0]).shape)
    print(np.array(solutions[task_id][0]))
    print('prediction:', np.array(prediction).shape)
    print(np.array(prediction))
    print('Score:', prediction == solutions[task_id][0])