
from typing import List, Tuple
import os
import json

def score_submission(submission_file_name, solutions) -> Tuple[float, int]:
    print('\n-----------------------------------------------------------------------')
    print (f"Scoring {submission_file_name}\n")

    with open(submission_file_name, "r") as file:
        submission = json.load(file)
    total_score = 0
    total_tasks = 0

    for task_id, task_submission in submission.items():
        total_tasks += 1
        task_score = 0
        num_pairs = len(task_submission)

        for pair_index, pair_attempts in enumerate(task_submission):
            # print(f"Scoring Task {task_id} pair #{pair_index+1}")
            pair_correct = False

            for attempt_key, attempt in pair_attempts.items():                
                if attempt == solutions[task_id][pair_index]:
                    print(f"- Task Id {task_id} pair {pair_index+1} {attempt_key} matches solution")
                    pair_correct = True
                    break # If it is correct, log it and break the loop

            if pair_correct:
                task_score += 1
            else:
                print(f"- Task Id {task_id} pair {pair_index+1} did not match solution")

        task_score /= num_pairs
        total_score += task_score

    return {
        'total_score': total_score,
        'total_tasks_scored': total_tasks
    }