from lp_ai.agents.base import setup_llm, setup_prompt
from lp_ai.output.parsing import check_output, parse_output, insert_errors
from lp_ai.graph.state import GraphState
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


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
    
    print(f'COMBINATOR Messages:')
    task_string = messages[0][1]
    ai_answers = ""
    for message in messages[1:]:
        ai_answers += f"-------------------------------------------------------------\n{message[1]}"
    print(ai_answers)

    if error == "yes":
        messages += [("user", "Now, try again. Invoke the code tool to structure the output with the patterns and test_output:",)]

    combinator_prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        """You are a VERY SMART AI called {llm_name} who is very good at solving puzzles. Below is a list of input and output pairs with a pattern.
        Your job is to look at the problem and compare the responses from the AIs (patterns already identified) and select the best patterns, analyze them and combine them. Also use your own knowledge about the problem.
        Hint: imagine the problem as a grid. Each number represents a different color. Imagine it visually and identify the pattern. Be very careful with the shape of the grids and identify the patterns for the inputs and outputs.
        Your GOAL is to apply the patterns and get the output for the test example. 
        INVOKE THE "CombineSolutionsTool" tool to structure the output correctly.
        """+"\n"+f"{ai_answers}",
        ),
        ("placeholder", "{messages}"),
    ]
)
    combinator_llm = setup_llm("llama3.1").with_structured_output(CombineSolutionsTool, include_raw=True) # gpt-4o

    # chain setup
    combinator_chain = combinator_prompt | combinator_llm | check_output
    # This will be run as a fallback chain
    N = 3
    fallback_chain = insert_errors | combinator_chain
    
    combinator_chain_retry = combinator_chain.with_fallbacks(fallbacks=[fallback_chain] * N, exception_key="error")
    combinator_chain = combinator_chain_retry | parse_output
    
    code_solution = combinator_chain.invoke(
        {"llm_name": "llama3", 
        "messages": [("user", task_string)]},
    )

    return {"generation": code_solution, "messages": [("assistant", code_solution)]}