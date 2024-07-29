import random
from lp_ai.agents.base import setup_llm, setup_prompt
from lp_ai.graph.state import GraphState
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List
from lp_ai.output.parsing import check_output, parse_output, insert_errors, extract_list


# Data model
class PatternsExtractionTool(BaseModel):
    model_name: str = Field(description="Name of the model that generated the patterns.")
    # patterns: List[str] = Field(description="List of patterns found in ALL the examples in the task (rules that apply from input to output in all examples).")
    patterns: str = Field(description="Patterns found in ALL the examples in the task (rules that apply from input to output in all examples).")
    description = "Schema for patterns identified in the challenge's task."


def generate_patterns(state: GraphState):
    messages = state["messages"]
    error = state["error"]

    task_string = messages[0][1]

    if error == "yes":
        messages += [("user", "Now, try again. Invoke the code tool to structure the output with a reasoning, imports, and code block:",)]

    temperature = random.uniform(0, 1)
    gen_llm = setup_llm("llama3.1", temperature).with_structured_output(PatternsExtractionTool, include_raw=True)

    gen_prompt = setup_prompt(
        """You are a VERY SMART AI called {llm_name} who is very good at solving puzzles. Below is a list of input and output pairs with a pattern.
        The objective is to identify 5 patterns or rules that can explain how to transform the input grid into the output grid for all provided examples.
        Objective:
        Analyze the given input-output examples and determine 5 patterns or rules that are consistent across all examples.
        Hint: imagine the problem as a grid. Each number represents a different color. Imagine it visually and identify the pattern. Be very careful with the shape of the grids and identify the patterns for the inputs and outputs.
        Use the "PatternsExtractionTool" tool to structure the output correctly based on the definitions.""", 
    )

    gen_chain = gen_prompt | gen_llm | check_output
    # # This will be run as a fallback chain
    fallback_chain = insert_errors | gen_chain
    N = 1
    gen_chain_retry = gen_chain.with_fallbacks(fallbacks=[fallback_chain] * N, exception_key="error")
    gen_chain = gen_chain_retry | parse_output

    print(f"---GENERATING PATTERNS llama3.1_{temperature}---")
    patterns = gen_chain.invoke(
        {"llm_name": f"llama3.1_{temperature}",
        "messages": [("user", task_string)]},
        )

    message = f"{patterns.model_name} Answer:\n{patterns.patterns}\n"
    return {"messages": [("assistant", message)]}