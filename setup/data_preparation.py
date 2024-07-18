import os
import json

data_prefix = 'arc-prize-2024'

# print ("Files included")
# for dirname, _, filenames in os.walk(data_prefix):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


task_sets = {
    'training' : {
        'challenges' : 'arc-prize-2024/arc-agi_training_challenges.json',
        'solutions' : 'arc-prize-2024/arc-agi_training_solutions.json',
    },
    'evaluation' : {
        'challenges' : 'arc-prize-2024/arc-agi_evaluation_challenges.json',
        'solutions' : 'arc-prize-2024/arc-agi_evaluation_solutions.json',
    }
}

def load_tasks_from_file(task_set):
    """
    Loads the tasks from the file and returns the challenges and solutions tasks
    """
    with open(task_set['challenges'], "r") as tasks:
        challenges = json.load(tasks)

    with open(task_set['solutions'], "r") as tasks:
        solutions = json.load(tasks)

    return challenges, solutions

challenges, solutions = load_tasks_from_file(task_set=task_sets['training'])


def json_task_to_string(challenge_tasks: dict, task_id: str, test_input_index: int) -> str:
    """
    challenge_tasks: dict a list of tasks
    task_id: str the id of the task we want to convert to a string
    
    Convert your json task into a string so you can pass it to your LLM.
    This is a crucial step where you can use your creativity to edit how tasks are represented.
    """
    json_task = challenge_tasks[task_id]

    final_output = "I will present a series of examples to you. Each example will have an input and an output. You will need to predict the output for the test example. You need to find the logical pattern for each input and the output for all the examples and follow the same logic to produce the output for the test case.\n\n"

    train_tasks = json_task['train']
    test_task = json_task['test']

    final_output += "Training Examples\n"

    for i, task in enumerate(train_tasks):
        final_output += f"Example {i + 1}: Input\n["
        for row in task['input']:
            final_output += f"\n{str(row)},"

        final_output += "]\n\n"
        final_output += f"Example {i + 1}: Output\n["

        for row in task['output']:
            final_output += f"\n{str(row)},"

        final_output += "]\n\n"

    final_output += "Test\n["
    for row in test_task[test_input_index]['input']:
        final_output += f"\n{str(row)}"

    final_output += "]\n\nYour Response:"

    return final_output