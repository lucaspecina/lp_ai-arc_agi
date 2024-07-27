import ast
import numpy as np

def check_output(tool_output):
    # print("\n----------------------------------------------\nTool Output:\n")
    # print(tool_output)
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
    print("Final Parsed Output:", solution["parsed"])
    return solution["parsed"]

def insert_errors(inputs):
    """Insert errors for tool parsing in the messages"""
    # Get errors
    error = inputs["error"]
    messages = inputs["messages"]
    messages += [
        (
            "assistant",
            f"Retry. You are required to fix the parsing errors: {error} \n\n You must invoke the provided tool.",
        )
    ]
    return {
        "messages": messages,
        "context": inputs["context"],
    }

def validate_output(test_task, solutions, task_id, test_output):
    prediction = ast.literal_eval(test_output)
    print('input:', test_task['input'])
    print('output:', np.array(solutions[task_id][0]).shape)
    print(np.array(solutions[task_id][0]))
    print('prediction:', np.array(prediction).shape)
    print(np.array(prediction))
    print('Score:', prediction == solutions[task_id][0])