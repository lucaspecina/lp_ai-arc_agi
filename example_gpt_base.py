from langchain_openai import ChatOpenAI
import argparse
import os

from setup.run import main

# Read API key from environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OpenAI API key is not defined.")

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Run the model with specified task set and number of tasks.")
    parser.add_argument('--task_set', type=str, default='training', help='Task set to use (e.g., training, testing, etc.)')
    parser.add_argument('--num_tasks', type=int, default=3, help='Number of tasks to run')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    # https://platform.openai.com/docs/models/models
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125', help='Model to use: \ngpt-3.5, gpt-3.5-turbo-0125, gpt-4, gpt-4-0125-preview, gpt-4-turbo')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # model setup
    if args.model == 'gpt-4':
        model_name = 'gpt4_base'
    elif args.model == 'gpt-3.5':
        model_name = 'gpt3_5_base'
    elif args.model == 'gpt-3.5-turbo-0125':
        model_name = 'gpt3_5_turbo_0125_base'
    elif args.model == 'gpt-4-0125-preview':
        model_name = 'gpt4_0125_preview_base'
    elif args.model == 'gpt-4-turbo':
        model_name = 'gpt4_turbo_base'
    else:
        raise ValueError(f"Invalid model: {args.model}")

    llm = ChatOpenAI(model='gpt-4', openai_api_key=openai_api_key, max_tokens=1000)

    if args.verbose:
        print(f'Testing {model_name}: "Describe en 20 palabras el ARC-AGI Challenge"')
        print(llm.invoke("Describe en 20 palabras el ARC-AGI Challenge").content)

    main(
        llm=llm,
        task_set= args.task_set,
        NUM_TASKS= args.num_tasks,
        submission_file_name= f'submissions/{model_name}-{args.task_set}-submission.json',
        verbose=args.verbose,
        )
