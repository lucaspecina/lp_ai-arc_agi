import random
from lp_ai.agents.base import setup_llm, setup_prompt, setup_chain
from lp_ai.graph.state import GraphState
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import List


# Output data model
class PromptingTool(BaseModel):
    gen_prompt: str = Field(description="Prompt for the pattern generators (with or without reflection).")
    description = "Schema for providing the prompt to the generators."


def agent_initiation(model, reflection=None, temperature=0.0):
    print(f'INITIATOR MODEL: {model}')

    init_prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        """
        You're a VERY SMART AI which is the BRAIN of a clever multi-LLM agent system built to solve puzzles.
        In this case, we're solving ARC-AGI, a set of puzzles that require understanding patterns and rules to transform input grids into output grids.
        Each ARC-AGI task consists of a set of input-output examples that follow a pattern.
        
        As the brain, you should be able to look at some particular task and tell the other AIs what to do in order to solve it.
        You have at your disposal a set of AIs (typically 5) that you should PROMPT in order for them to generate patterns. 
        The prompt will be the same for all of them, so you should be able to guide them to generate the non-trivial patterns that solve the task.
        
        You need to UNDERSTAND the problem and create the most effective prompt to guide the pattern generators to solve the ARC-AGI tasks.
        Use the "PromptingTool" tool to structure the output correctly based on the description.
        Explain the problem extensively and give context but don't include the inputs or outputs in the prompt, just the instructions to the AIs (the task will be provided to them).
        Tell them that they should produce some non-trivial patterns that can be used to solve the task and list them.

        Below is the task you need to help the AIs solve:""",
        ),
        ("placeholder", "{messages}"),
    ]
    )
    # LLM setup
    init_llm = setup_llm(
        model_name=model, 
        temperature=temperature, 
        max_tokens=3000, 
        tools=PromptingTool, 
    )
    # chain setup
    init_chain = setup_chain(init_prompt, init_llm, retries=3)
    
    return init_chain

# gen_chain = agent_generate_patterns(0.3)

def node_initiate(state: GraphState, config):
    print("\n\n------INITIATING SYSTEM AND GENERATING PROMPTS------")
    init_model = config["configurable"]["initiator_model"]

    # TODO: take the messages and reflection from the state
    messages = state["messages"]
    error = state["error"]
    iterations = state["iterations"]
    current_messages = []

    task_string = messages[0][1]
    current_messages += [("user", task_string)]

    # TODO: create the context for reflection using past messages and evaluation
    # We have been routed back to generation with an error
    if error == "yes":
        messages += [("user", "Now, try again. Invoke the code tool to structure the output:",)]
        current_messages += [("user", "Now, try again. Invoke the code tool to structure the output:",)]
        print("Error in the previous step. Try again.")

    # chain setup
    init_chain = agent_initiation(
                                init_model, 
                                reflection=None, 
                                temperature=0.3
                                )
    
    # Invoke graph
    init_prompt = init_chain.invoke(
        {"llm_name": init_model,
        "messages": messages},
        )
    # Increment
    iterations = iterations + 1

    message = f"Initial prompt: \n{init_prompt.gen_prompt}"
    print(f"INITIATOR PROMPT: {message}\n\n")
    return {"messages": [("user", message)], 
            # "error": state['error'], 
            # "iterations": iterations
            }