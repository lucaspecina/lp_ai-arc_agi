from typing import List, Tuple
import os
import json
import random

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

from setup.output_parsing import parse_output
from setup.data_preparation import json_task_to_string


class ARCPrediction(BaseModel):
    prediction: List[List] = Field(..., description="A prediction for a task")


def get_task_prediction(llm, challenge_tasks, task_id, test_input_index, verbose=False) -> List[List]:
    task_string = json_task_to_string(challenge_tasks, task_id, test_input_index)    
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
    chain = prompt | llm | parser
    if verbose:
        print (f"Prompt:\n\n{prompt.format(task_string=task_string)}")

    output = chain.invoke({"task_string": task_string})
    if verbose:
        print(output)

    prediction = parse_output(output, verbose=verbose)
    return prediction


def run_model(llm, challenges, NUM_ATTEMPTS=2, RETRY_ATTEMPTS=3, NUM_TASKS=None, verbose=False):
    submission = {}
    # random.seed(42)

    shuffled_task_ids = list(challenges.keys())
    random.shuffle(shuffled_task_ids)

    for i, task_id in enumerate(shuffled_task_ids):
        task_attempts = []

        # Go through each test pair to get a prediction. 96% of challenges have 1 pair.
        for t, pair in enumerate(challenges[task_id]['test']):
            print('\n-----------------------------------------------------------------------')
            print(f"Predicting task #{i + 1} ({task_id}), pair #{t+1}")
            print('---')
            pair_attempts = {}  

            for attempt in range(1, NUM_ATTEMPTS + 1):
                attempt_key = f"attempt_{attempt}"
                pair_attempts[attempt_key] = []

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
                        
                        pair_attempts[attempt_key] = prediction
                        break  # Break the retry loop if prediction is successful
                    except Exception as e:
                        print(f"Retrying: {e}")
                        if retry == RETRY_ATTEMPTS - 1:
                            pair_attempts[attempt_key] = []  # None if all retries fail
            
            task_attempts.append(pair_attempts)
        submission[task_id] = task_attempts

        if NUM_TASKS is not None and i + 1 == NUM_TASKS:
            break

    return submission
