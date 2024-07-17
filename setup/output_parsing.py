from langchain_core.output_parsers import JsonOutputParser # To help with structured output
from langchain_core.prompts import PromptTemplate # To help create our prompt
from langchain_core.pydantic_v1 import BaseModel, Field # To help with defining what output structure we want

from typing import List, Tuple
import os
import json
from setup.data_preparation import json_task_to_string

# Defining a prediction as a list of lists
class ARCPrediction(BaseModel):
    prediction: List[List] = Field(..., description="A prediction for a task")

def get_task_prediction(llm, challenge_tasks, task_id, test_input_index, verbose=False) -> List[List]:
    """
    challenge_tasks: dict a list of tasks
    task_id: str the id of the task we want to get a prediction for
    test_input_index: the index of your test input. 96% of tests only have 1 input.

    Given a task, predict the test output
    """

    # Get the string representation of your task
    task_string = json_task_to_string(challenge_tasks, task_id, test_input_index)
    
    # Set up a parser to inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=ARCPrediction)

    # Create your prompt template. This is very rudimentary! You should edit this to do much better.
    # For example, we don't tell the model what it's first attempt was (so it can do a different one), that might help!
    prompt = PromptTemplate(
        template="You are a bot that is very good at solving puzzles. Below is a list of input and output pairs with a pattern." 
                    "Identify the pattern, then apply that pattern to the test input to give a final output"
                    "Just give valid json list of lists response back, nothing else. Do not explain your thoughts."
                    "{format_instructions}\n{task_string}\n",
        input_variables=["task_string"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Wrap up your chain with LCEL
    chain = prompt | llm | parser

    # Optional, print out the prompt if you want to see it. If you use LangSmith you could view this there as well.
    if verbose:
        print (f"Prompt:\n\n{prompt.format(task_string=task_string)}")
    
    # Finally, go get your prediction from your LLM. Ths will make the API call.
    output = chain.invoke({"task_string": task_string})

    # Because the output is structured, get the prediction key. If it isn't there, then just get the output
    if isinstance(output, dict):
        prediction = output.get('prediction', output)
    else:
        prediction = output

    # TODO: add checks for the shape. It should be equal to the test output shape.
    # Safety measure to error out if you don't get a list of lists of ints back. This will spark a retry later.
    if not all(isinstance(sublist, list) and all(isinstance(item, int) for item in sublist) for sublist in prediction):
        if verbose:
            print("Warning: Output must be a list of lists of integers.")
            print (f"Errored Output: {prediction}")
        raise ValueError("Failed: Output must be a list of lists of integers.")
    else:
        print('Success')
    
    # Let's find the shape of our prediction
    num_rows = len(prediction)
    num_cols = len(prediction[0]) if num_rows > 0 else 0
    
    if verbose:
        print(f"Prediction Grid Size: {num_rows}x{num_cols}\n")
        print(f"Prediction: \n{prediction}")
    
    return prediction


def create_submission_file(submission, file_name='submissions/submission.json'):
    """
    Save a submission file to the specified file name
    """
    with open(file_name, "w") as file:
        json.dump(submission, file)
    
    print('\n-----------------------------------------------------------------------')
    print (f"Submission saved to {file_name}")