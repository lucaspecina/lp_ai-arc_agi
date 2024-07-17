
from setup.model_inference import run_model
from setup.data_preparation import load_tasks_from_file, task_sets
from setup.scoring import score_submission
from setup.output_parsing import create_submission_file


def main(llm, task_set='training', NUM_TASKS=None, submission_file_name='submissions/submission.json', verbose=False):
    # Load datasets
    challenges, solutions = load_tasks_from_file(task_set=task_sets[task_set])

    # Run the model
    submission = run_model(llm, challenges, NUM_TASKS=NUM_TASKS, verbose=verbose)

    # Create (and overwrite) a submission file
    create_submission_file(submission, file_name=submission_file_name)

    # Score the submission
    score_result = score_submission(solutions = solutions, submission_file_name=submission_file_name)

    print('\n-----------------------------------------------------------------------')
    print(f"Final score: {score_result['total_score']} of {score_result['total_tasks_scored']} ({round(score_result['total_score']/score_result['total_tasks_scored'] * 100, 2)}%)")
