import random
from lp_ai.agents.base import setup_llm, setup_prompt, setup_chain
from lp_ai.graph.state import GraphState
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import List


# Output data model
class PatternsExtractionTool(BaseModel):
    model_name: str = Field(description="Name of the model that generated the patterns.")
    patterns: str = Field(description="Patterns found in ALL the examples in the task (non-trivial rules that apply from input to output in all examples).")
    description = "Schema for patterns identified in the challenge's task. Don't use brackets {} in the responses."

# TODO: create classes for the agents (base and specific)
def agent_generate_patterns(init_prompt, temperature=0.0):

    gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        f"""
        {init_prompt}
        
        Use the "PatternsExtractionTool" tool to structure the output correctly based on the description.
        The transformation rules should be such that if applied to the training input examples, they should produce the training output examples.
        
        Below is the task:""",
        ),
        ("placeholder", "{messages}"),
    ]
    )
    # LLM setup
    gen_llm = setup_llm(
        model_name="llama3.1", 
        temperature=temperature, 
        max_tokens=1000, 
        tools=PatternsExtractionTool, 
    )
    # chain setup
    gen_chain = setup_chain(gen_prompt, gen_llm)
    
    return gen_chain


def node_generate_patterns(state: GraphState):
    
    task_string = state["task_string"] # messages[0][1]
    messages = state["messages"]
    error = state["error"]
    init_prompt = state["gen_prompt"] # init_prompt = messages[-1][1]
    iterations = state["iterations"]
    max_reflections = state["max_reflections"]

    # We have been routed back to generation with an error
    if error == "yes":
        messages += [("user", "Now, try again. Invoke the code tool to structure the output:",)]
        print("Error in the previous step. Try again.")

    # chain setup
    temperature = random.uniform(0, 0.5)
    gen_chain = agent_generate_patterns(init_prompt, temperature)
    
    # Invoke graph
    print(f"------[{iterations+1}/{max_reflections} reflections] GENERATING PATTERNS llama3.1_{temperature}------")
    patterns = gen_chain.invoke(
        {"llm_name": f"llama3.1_{temperature}",
        "messages": [("user", task_string)]},
        )

    message = f"{patterns.model_name} Answer:\n{patterns.patterns}\n"
    return {"messages": [("assistant", message)], }