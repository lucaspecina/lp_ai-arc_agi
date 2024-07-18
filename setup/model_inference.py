from typing import List, Tuple
import os
import json
import random

from langchain_core.output_parsers import JsonOutputParser # To help with structured output
from langchain_core.prompts import PromptTemplate # To help create our prompt
from langchain_core.pydantic_v1 import BaseModel, Field # To help with defining what output structure we want

from setup.output_parsing import parse_output
from setup.data_preparation import json_task_to_string


class ARCPrediction(BaseModel):
    prediction: List[List] = Field(..., description="A prediction for a task")


def get_task_prediction(llm, challenge_tasks, task_id, test_input_index, verbose=False) -> List[List]:

    # Get the string representation of your task
    task_string = json_task_to_string(challenge_tasks, task_id, test_input_index)
    
    # Set up a parser to inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=ARCPrediction)

    # TODO: add chain of thought capability (should also parse the output in a correct way)
    # TODO: tell the model what it's first attempt was (so it can do a different one)
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

    if verbose:
        print(output)

    # Parse the output
    prediction = parse_output(output, verbose=verbose)

    return prediction


def run_model(llm, challenges, NUM_ATTEMPTS=2, RETRY_ATTEMPTS=3, NUM_TASKS=None, verbose=False):
    """
    challenges: dict a list of challenges. This should come directly from your _challenges file
    NUM_ATTEMPTS: int the number of times to attempt a prediction. The official competition has 2 attempts.
    RETRY_ATTEMPTS: int the number of times to retry a prediction if it fails
    NUM_TASKS: int, If set, this represents the the number of tasks you'd like to test. If None then the all challeneges will be tested

    Loop through your challenges and produce a submission.json file you can submit for a score.
    """

    # A dict to hold your submissions that you'll return after all predictions are made
    submission = {}

    # Set the seed for reproducibility
    # random.seed(42)

    # Shuffle the task IDs
    shuffled_task_ids = list(challenges.keys())
    random.shuffle(shuffled_task_ids)

    # Run through each task in the shuffled order
    for i, task_id in enumerate(shuffled_task_ids):
        task_attempts = []

        # Go through each test pair to get a prediction. 96% of challenges have 1 pair.
        for t, pair in enumerate(challenges[task_id]['test']):
            print('\n-----------------------------------------------------------------------')
            print(f"Predicting task #{i + 1} ({task_id}), pair #{t+1}")
            print('---')

            # Dictionary to store attempts for the current test pair
            pair_attempts = {}  

            # Run through each prediction attempt
            for attempt in range(1, NUM_ATTEMPTS + 1):
                attempt_key = f"attempt_{attempt}"
                pair_attempts[attempt_key] = [] # Init your attempt

                # Try to get a prediction, with retries in case of failure
                for retry in range(RETRY_ATTEMPTS):
                    try:
                        print(f"\nPredicting attempt #{attempt}, retry #{retry}")
                        prediction = get_task_prediction(llm=llm,
                                                         challenge_tasks=challenges,
                                                         task_id=task_id,
                                                         test_input_index=t,
                                                         verbose=verbose
                                                         )
                        
                        # If you get a valid prediction (list of lists of ints) with no error, then log the attempt
                        pair_attempts[attempt_key] = prediction
                        break  # Break the retry loop if prediction is successful
                    except Exception as e:
                        print(f"Retrying: {e}")
                        if retry == RETRY_ATTEMPTS - 1:
                            pair_attempts[attempt_key] = []  # Assign None if all retries fail

            # After you get your attempts, append them to the task attempts
            task_attempts.append(pair_attempts)

        # Append the task attempts to the submission with the task_id as the key
        submission[task_id] = task_attempts

        # If you want to stop after N tasks, uncomment the below
        if NUM_TASKS is not None and i + 1 == NUM_TASKS:
            break

    return submission
