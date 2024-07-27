from lp_ai.agents.base import setup_llm, setup_prompt
from lp_ai.output.parsing import check_output, parse_output, insert_errors
from lp_ai.graph.state import GraphState
from langchain_core.pydantic_v1 import BaseModel, Field

# TOOL
# Data model
class CombineSolutionsTool(BaseModel):
    # reasoning: str = Field(description="Chain of thought process explaining step by step the reasoning of the solution.")
    patterns: str = Field(description="Enumerate the patterns used to solve the problem.")
    # imports: str = Field(description="Code block import statements")
    # code: str = Field(description="Code block not including import statements")
    test_output: str = Field(description="Output for the TEST case (applying the patterns).")
    description = "Schema for code solutions to questions about the challenge."


def combination(state: GraphState):
    print("---COMBINING CODE SOLUTION---")

    messages = state["messages"]
    error = state["error"]
    print(f'COMBINATOR\nMessages:')
    for message in messages[1:]:
        print('-------------------------------------------------------------')
        print(message[1])

    if error == "yes":
        messages += [("user", "Now, try again. Invoke the code tool to structure the output with a reasoning, imports, and code block:",)]

    combinator_prompt = setup_prompt(
        """You are a VERY SMART AI called {llm_name} who is very good at solving puzzles. Below is a list of input and output pairs with a pattern.
        Your job is to look at the problem and compare the responses from the AIs (patterns already identified) and select the best patterns, analyze them and combine them. Also use your own knowledge about the problem.
        Hint: imagine the problem as a grid. Each number represents a different color. Imagine it visually and identify the pattern. Be very careful with the shape of the grids and identify the patterns for the inputs and outputs.
        Your GOAL is to apply the patterns and get the output for the test example. INVOKE THE "CombineSolutionsTool" tool to structure the output correctly based on the definitions.""",
        # messages
    )
    combinator_llm = setup_llm("gpt-4o").with_structured_output(CombineSolutionsTool, include_raw=True)

    # build chain
    combinator_chain = combinator_prompt | combinator_llm | check_output
    # # This will be run as a fallback chain
    fallback_chain = insert_errors | combinator_chain
    N = 1
    combinator_chain_retry = combinator_chain.with_fallbacks(fallbacks=[fallback_chain] * N, exception_key="error")
    combinator_chain = combinator_chain_retry | parse_output
    
    code_solution = combinator_chain.invoke({"llm_name": "COMBINATOR", "messages": messages})

    return {"generation": code_solution, "messages": [("assistant", code_solution)]}