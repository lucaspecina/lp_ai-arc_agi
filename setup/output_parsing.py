import json


def parse_output(output, verbose=False):
    """
    Parse the output from the model
    """
    # TODO: add functionality for Chain of Thought -> should be able to generate reasoning steps 
    # and after a key word (answer) say the answer in the right format (see implementations of CoT)
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