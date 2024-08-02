from typing import Annotated, TypedDict, List
import operator

class GraphState(TypedDict):
    task_id: str
    task_string: str
    n_generators: int
    max_reflections: int
    error: str
    messages: Annotated[list, operator.add]
    iterations: int
    gen_prompt: str
    rules: str
    training_predictions: List[str]
    test_output: str
    feedback: str