from typing import Annotated
from typing import TypedDict
import operator

class GraphState(TypedDict):
    error: str
    messages: Annotated[list, operator.add]
    iterations: int
    generation: str
    evaluation: str