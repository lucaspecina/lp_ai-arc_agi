import os
from lp_ai.data.data_processing import load_tasks_from_file, task_sets
from lp_ai.graph.workflow import setup_workflow
from lp_ai.data.data_processing import json_task_to_string
from lp_ai.output.scoring import test_task_multiple
import argparse


def main(task_id, num_generators, max_reflections, rounds, initiator_model, combinator_model, evaluator_model, debug=False):
    
    # Load task
    challenges, solutions = load_tasks_from_file(task_sets['training'])
    task_string = json_task_to_string(challenges, task_id, 0)
    print(f"Task string:\n{task_string}") if debug else None

    # Setup graph workflow
    app = setup_workflow(num_generators)

    # Invoke graph
    final_answers = []
    for i in range(rounds):
        print(f"\n\nROUND {i+1}/{rounds}"+"-"*40+"\n")
        output = app.invoke(
            {"messages": [("user", task_string)],
             "task_id": task_id,
             "iterations": 0, 
             "n_generators": num_generators,
             "task_string": task_string,
             "max_reflections": max_reflections,}, 
            {"configurable": {
                "initiator_model": initiator_model, 
                "combinator_model": combinator_model, 
                "evaluator_model": evaluator_model}
                },
            debug=False,
            )
        test_output = output['test_output']

        if debug:
            print(f"Task ID: {task_id}")
            print("\n\nGenerated Solution:\n")
            print(f"Patterns used: \n{output['rules']}\n")
            print(f"Final answer:")
            print(test_output)
        final_answers.append(test_output)

    # Display results
    test_task_multiple(final_answers, challenges, solutions, task_id)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--debug", action="store_true", help="Enable verbose output for debugging", default=False)
    parser.add_argument("-t", "--task_id", type=str, help="Task ID", default="0520fde7")
    parser.add_argument("-n", "--num_generators", type=int, help="Number of generators", default=3)
    parser.add_argument("-m", "--max_reflections", type=int, help="Max iterations for reflection", default=3)
    parser.add_argument("-r", "--rounds", type=int, help="Rounds for trying solutions", default=1)
    parser.add_argument("-i", "--initiator_model", type=str, help="Initiator Model", default="llama3.1")
    parser.add_argument("-c", "--combinator_model", type=str, help="Combinator Model", default="llama3.1")
    parser.add_argument("-e", "--evaluator_model", type=str, help="Evaluator Model", default="llama3.1")
    args = parser.parse_args()

    print(f"Task ID: {args.task_id}")
    print(f"Number of generators: {args.num_generators}")
    print(f"Debug: {args.debug}")
    print(f"Max reflections: {args.max_reflections}")
    print(f"Rounds: {args.rounds}")

    main(
        args.task_id, 
        args.num_generators, 
        args.max_reflections, 
        args.rounds,
        args.initiator_model, 
        args.combinator_model, 
        args.evaluator_model, 
        args.debug
        )
