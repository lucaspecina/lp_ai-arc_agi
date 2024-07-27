import os
import sys
import numpy as np
from collections import Counter
import ast
from lp_ai.data.data_processing import load_tasks_from_file, task_sets
from lp_ai.graph.workflow import setup_workflow
from lp_ai.data.data_processing import json_task_to_string
import argparse
from collections import Counter


def main(verbose, task_id, num_generators, num_iterations):
    
    # Load task
    challenges, solutions = load_tasks_from_file(task_sets['training'])
    task_string = json_task_to_string(challenges, task_id, 0)
    # print("Task string:")
    # print(task_string)

    # Setup graph workflow
    app = setup_workflow(num_generators)

    # Invoke graph
    final_answers = []
    for i in range(num_iterations):
        print(f"\n\nITERATION {i}\n\n")
        result = app.invoke({"messages": [("user", task_string)]}, debug=False)
        test_output = result['generation'].test_output
        # Print output
        if verbose:
            print(f"Task ID: {task_id}")
            print("\n\nGenerated Solution:\n")
            print(f"Reasoning: \n{result['generation'].reasoning}\n")
            print(f"Patterns used: \n{result['generation'].patterns}\n")
            print(f"Description: \n{result['generation'].description}\n")    
            print(f"Final answer:")
            print(test_output)
        final_answers.append(test_output)
    # result = app.invoke({"messages": [("user", task_string)]}, debug=verbose)
    # test_output = result['generation'].test_output

    # Test example
    # MOST REPEATED ANSWER
    print('\n--------------------------------- MOST REPEATED ANSWER ---------------------------------')
    answers_count = Counter(final_answers)
    print(f"Task ID: {task_id}")
    print('\nTEST EXAMPLE:')
    test_task = challenges[task_id]['test'][0]
    prediction = ast.literal_eval(answers_count.most_common(1)[0][0])
    print('input:', test_task['input'])
    print('output:', np.array(solutions[task_id][0]).shape)
    print(np.array(solutions[task_id][0]))
    print('prediction:', np.array(prediction).shape)
    print(np.array(prediction))
    print('Score:', prediction == solutions[task_id][0])

    # ALL ANSWERS
    print('\n--------------------------------- ALL ANSWERS ---------------------------------')
    print(f"Task ID: {task_id}")
    print('\nTEST EXAMPLE:')
    print('input:', test_task['input'])
    print('output:', np.array(solutions[task_id][0]).shape)
    print(np.array(solutions[task_id][0]))

    for i, answer in enumerate(final_answers):
        try:
            prediction = ast.literal_eval(answer)
            print(f'Score answer {i+1}:', prediction == solutions[task_id][0])
            if prediction == solutions[task_id][0]:
                print('prediction:', np.array(prediction).shape)
                print(np.array(prediction))
        except:
            print(f"Answer {i+1}: Bad format")
            print(answer)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output", default=False)
    parser.add_argument("-t", "--task_id", type=str, help="Task ID", default="0520fde7")
    parser.add_argument("-n", "--num_generators", type=int, help="Number of generators", default=3)
    parser.add_argument("-i", "--num_iterations", type=int, help="Number of generators", default=1)
    args = parser.parse_args()

    print(f"Task ID: {args.task_id}")
    print(f"Number of generators: {args.num_generators}")
    print(f"Verbose: {args.verbose}")
    print(f"Number of iterations: {args.num_iterations}")
    main(args.verbose, args.task_id, args.num_generators, args.num_iterations)
