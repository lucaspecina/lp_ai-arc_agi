from langchain_openai import ChatOpenAI
import argparse
import os

from setup.run import main

# Read API key from environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("OpenAI API key is not defined.")

model_name = 'gpt4_base'

llm = ChatOpenAI(model='gpt-4', openai_api_key=openai_api_key, max_tokens=1000)

print(f'Testing {model_name}: "Describe en 20 palabras el ARC-AGI Challenge"')
print(llm.invoke("Describe en 20 palabras el ARC-AGI Challenge").content)

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Run the model with specified task set and number of tasks.")
    parser.add_argument('--task_set', type=str, default='training', help='Task set to use (e.g., training, testing, etc.)')
    parser.add_argument('--num_tasks', type=int, default=3, help='Number of tasks to run')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    main(
        llm=llm,
        task_set= args.task_set,
        NUM_TASKS= args.num_tasks,
        submission_file_name= f'submissions/{model_name}-{args.task_set}-submission.json',
        verbose=args.verbose,
        )
