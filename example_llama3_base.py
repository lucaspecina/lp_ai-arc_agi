from langchain_community.llms import Ollama
import argparse

from setup.run import main

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Run the model with specified task set and number of tasks.")
    parser.add_argument('--task_set', type=str, default='training', help='Task set to use (e.g., training, testing, etc.)')
    parser.add_argument('--num_tasks', type=int, default=3, help='Number of tasks to run')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--model', type=str, default='llama3', help='Model to use (e.g., llama3, llama3:70b)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # model setup
    if args.model == 'llama3':
        model_name = 'llama3_8b_base'
    elif args.model == 'llama3:70b':
        model_name = 'llama3_70b_base'
    else:
        raise ValueError(f"Invalid model: {args.model}")
    
    llm = Ollama(model=args.model)
    
    if args.verbose:
        print(f'Testing {model_name}: "Describe en 20 palabras el ARC-AGI Challenge"')
        print(llm.invoke("Describe en 20 palabras el ARC-AGI Challenge"))

    main(
        llm=llm,
        task_set= args.task_set,
        NUM_TASKS= args.num_tasks,
        submission_file_name= f'submissions/{model_name}-{args.task_set}-submission.json',
        verbose=args.verbose,
        )
