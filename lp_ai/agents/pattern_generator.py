import random
from lp_ai.agents.base import setup_llm, setup_prompt, setup_chain
from lp_ai.graph.state import GraphState
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List


# Output data model
class PatternsExtractionTool(BaseModel):
    model_name: str = Field(description="Name of the model that generated the patterns.")
    patterns: str = Field(description="Patterns found in ALL the examples in the task (rules that apply from input to output in all examples).")
    description = "Schema for patterns identified in the challenge's task."

# TODO: create classes for the agents (base and specific)
def agent_generate_patterns(temperature=0.0):

    gen_prompt = setup_prompt(
        """You are a VERY SMART AI called {llm_name} who is very good at solving puzzles. Below is a list of input and output pairs with a pattern.
        The objective is to identify 5 patterns or rules that can explain how to transform the input grid into the output grid for all provided examples.
        Objective:
        Analyze the given input-output examples and determine 5 patterns or rules that are consistent across all examples.
        Hint: imagine the problem as a grid. Each number represents a different color. Imagine it visually and identify the pattern. Be very careful with the shape of the grids and identify the patterns for the inputs and outputs.
        Use the "PatternsExtractionTool" tool to structure the output correctly based on the definitions.""", 
    )
    # LLM setup
    gen_llm = setup_llm(
        model_name="llama3.1", 
        temperature=temperature, 
        max_tokens=1000, 
        tools=PatternsExtractionTool, 
    )
    # chain setup
    gen_chain = setup_chain(gen_prompt, gen_llm, retries=3)
    
    return gen_chain

# gen_chain = agent_generate_patterns(0.3)

def node_generate_patterns(state: GraphState):
    messages = state["messages"]
    error = state["error"]
    iterations = state["iterations"]
    current_messages = []

    task_string = messages[0][1]
    current_messages += [("user", task_string)]

    # We have been routed back to generation with an error
    if error == "yes":
        messages += [("user", "Now, try again. Invoke the code tool to structure the output:",)]
        current_messages += [("user", "Now, try again. Invoke the code tool to structure the output:",)]
        print("Error in the previous step. Try again.")

    # chain setup
    temperature = random.uniform(0, 0.3)
    gen_chain = agent_generate_patterns(temperature)
    
    # Invoke graph
    print(f"------GENERATING PATTERNS llama3.1_{temperature}------")
    patterns = gen_chain.invoke(
        {"llm_name": f"llama3.1_{temperature}",
        # "messages": current_messages},
        "messages": messages},
        )
    # Increment
    iterations = iterations + 1

    message = f"{patterns.model_name} Answer:\n{patterns.patterns}\n"
    return {"messages": [("assistant", message)], 
            # "error": state['error'], 
            # "iterations": iterations
            }