from lp_ai.agents.base import setup_llm, setup_prompt, setup_chain
from lp_ai.output.parsing import check_output, parse_output, insert_errors
from lp_ai.graph.state import GraphState
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# TOOL
# Data model
class CombinePatternsTool(BaseModel):
    patterns: str = Field(description="Enumerate the patterns used to solve the problem.")
    test_output: str = Field(description="Output for the TEST case (applying the patterns).")
    description = "Schema for code solutions to questions about the challenge."


def agent_combine_patterns(ai_answers, temperature=0.0):

    combinator_prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        """You are a VERY SMART AI called {llm_name} who is very good at solving puzzles. Below is a list of input and output pairs with a pattern.
        Your job is to look at the problem and compare the responses from the AIs (patterns already identified) and select the best patterns, analyze them and combine them. Also use your own knowledge about the problem.
        Hint: imagine the problem as a grid. Each number represents a different color. Imagine it visually and identify the pattern. Be very careful with the shape of the grids and identify the patterns for the inputs and outputs.
        Your GOAL is to apply the patterns and get the output for the test example. 
        INVOKE THE "CombinePatternsTool" tool to structure the output correctly.
        """+"\n"+f"{ai_answers}",
        ),
        ("placeholder", "{messages}"),
    ]
    )   
    # LLM setup
    combinator_llm = setup_llm(
        model_name="llama3.1", 
        temperature=temperature, 
        max_tokens=1000, 
        tools=CombinePatternsTool, 
    )
    # chain setup
    combinator_chain = setup_chain(combinator_prompt, combinator_llm, retries=3)
    
    return combinator_chain


def node_combine_patterns(state: GraphState):
    print("---COMBINING PATTERNS AND GENERATING FINAL SOLUTION---")

    messages = state["messages"]
    error = state["error"]
    iterations = state["iterations"]
    
    print(f'COMBINATOR Messages:')
    task_string = messages[0][1]
    ai_answers = ""
    for message in messages[1:]:
        ai_answers += f"-------------------------------------------------------------\n{message[1]}"
    print(ai_answers)

    # We have been routed back to generation with an error
    if error == "yes":
        messages += [("user", "Now, try again. Invoke the code tool to structure the output with the patterns and test_output:",)]

    # chain setup
    combinator_chain = agent_combine_patterns(ai_answers)
    
    # Invoke graph
    print(f"---GENERATING FINAL SOLUTION llama3.1---")
    final_solution = combinator_chain.invoke(
        {"llm_name": "llama3.1", 
        "messages": [("user", task_string)]},
    )
    # Increment
    iterations = iterations + 1

    return {"generation": final_solution, 
            "messages": [("assistant", final_solution)],
            "error": state['error'],
            "iterations": iterations
            }