from typing import Annotated
from typing import TypedDict
import operator

class GraphState(TypedDict):
    task_string: str
    n_generators: int
    max_reflections: int
    error: str
    messages: Annotated[list, operator.add]
    iterations: int
    gen_prompt: str
    rules: str
    test_output: str
    feedback: str