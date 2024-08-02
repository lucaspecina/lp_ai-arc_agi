import numpy as np 
from collections import Counter
import ast
from lp_ai.data.data_processing import load_tasks_from_file, task_sets
import json


def parse_final_output(task_id, predictions):
    submission = {}
    submission[task_id] = []
    
    # Predictions should be two
    if len(predictions) != 2:
        raise ValueError(f"Failed: Output must be a list of two lists of integers.\n{predictions}")
    
    test_outputs = {}

    for i, prediction in enumerate(predictions):
        # Safety measure to error out if you don't get a list of lists of ints back. This will spark a retry later.
        if not all(isinstance(sublist, list) and all(isinstance(item, int) for item in sublist) for sublist in prediction):
            raise ValueError(f"Failed: Output must be a list of lists of integers.\n{prediction}")
    
        test_outputs[f"attempt_{i+1}"] = prediction

    submission[task_id][0] = test_outputs
    
    return submission


def create_submission_file(submission, file_name='submissions/submission.json'):
    with open(file_name, "w") as file:
        json.dump(submission, file)
    
    print('\n-----------------------------------------------------------------------')
    print (f"Submission saved to {file_name}")


def test_training_examples(training_predictions, task_id):
    """Compares the training_predictions with the real training outputs for a task."""
    challenges, solutions = load_tasks_from_file(task_sets['training'])

    assert len(training_predictions) == len(challenges[task_id]['train']), "Number of training examples and predictions do not match"
    
    training_examples = []
    for i, train_task in enumerate(challenges[task_id]['train']):
        try:
            training_examples.append({
                "example": i+1, 
                "input": train_task['input'], 
                "output": train_task['output'], 
                "prediction": ast.literal_eval(training_predictions[i]), 
                "score": ast.literal_eval(training_predictions[i]) == train_task['output']})
        except:
            training_examples.append({
                "example": i+1, 
                "input": train_task['input'], 
                "output": train_task['output'], 
                "prediction": "Bad format", 
                "score": False})
    return training_examples


def test_individual_task(gen_code, challenges, solutions, task_id):
    code_namespace = {}
    
    # Execute the code to define the function in the local namespace
    exec(gen_code, code_namespace)
    
    # Extract the function from the local namespace
    solve = code_namespace['solve']
    print(solve)
    
    print('TRAIN EXAMPLES:')
    for i, train_task in enumerate(challenges[task_id]['train']):
        
        # Get the input for the current train task
        input_grid = train_task['input']
        
        # Call the function with the input
        prediction = solve(input_grid)
        
        print(f"\nTrain Task {i+1}")
        print('input:', train_task['input'])
        print('output:', train_task['output'])
        print('prediction:', prediction)
        print('Score:', prediction == train_task['output'])

    print('\nTEST EXAMPLE:')
    test_task = challenges[task_id]['test'][0]
    
    # Call the function with the input for the test task
    prediction = solve(test_task['input'])
    
    print('input:', test_task['input'])
    print('output:', np.array(solutions[task_id][0]).shape)
    print(np.array(solutions[task_id][0]))
    print('prediction:', np.array(prediction).shape)
    print(np.array(prediction))
    print('Score:', prediction == solutions[task_id][0])
    
    return solve

def test_task_multiple(final_answers, challenges, solutions, task_id):
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
            print(np.array(prediction))
        except:
            print(f"Answer {i+1}: Bad format")
            print(answer)
    
    return ast.literal_eval(answers_count.most_common(1)[0][0])
