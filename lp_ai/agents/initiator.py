import random
from lp_ai.agents.base import setup_llm, setup_prompt, setup_chain
from lp_ai.graph.state import GraphState
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from typing import List
import langchain 

# Output data model
class PromptingTool(BaseModel):
    gen_prompt: str = Field(description="Prompt for the pattern generators (with or without reflection).")
    description = "Schema for providing the prompt to the generators. Don't use brackets {} in the responses."


def agent_initiation(model, feedback=None, rules=None, temperature=0.0):
    print(f'INITIATOR MODEL: {model}')

    base_init_prompt = """You're a VERY SMART AI which is the BRAIN of a clever multi-LLM agent system built to solve puzzles.
        In this case, we're solving ARC-AGI, a set of puzzles that require understanding patterns and rules to transform input grids into output grids.
        Each ARC-AGI task consists of a set of input-output examples that follow a pattern (a rule of transformation from input to output).
        THE GOAL OF THE WHOLE SYSTEM IS TO DISCOVER WHAT THOSE TRANSFORMATION RULES ARE (they should work for each example in the task).
        
        As the brain, you will be looking at some particular task (set of input-output pairs with the same transformation rule) and tell some other AIs what to do in order to solve it.
        You have at your disposal a set of AIs (typically 5) that you should PROMPT in order for them to generate a list of possible transformation rules. 
        The prompt will be the same for all of the AIs, so you should be able to guide them to generate the non-trivial transformation rules that solve the task.
        
        You need to UNDERSTAND the problem and create the most effective prompt to guide the pattern generators to solve the ARC-AGI task.
        Explain the problem extensively and give context but don't include the inputs or outputs in the prompt, just the instructions to the AIs (the task will be provided to them).
        Tell them that they should produce 5 non-trivial patterns that can be used to solve the task and list them. They don't need to generate the outputs, only LIST THE RULES.
        """
    
    if feedback is not None:
        base_init_prompt += f"""\n--------------------------------------------------
        REFLECTION: 
        There has been already previous attempts to solve the task but the evaluator found that the patterns/rules generated were not accurate enough.
        --------------------------------------------------
        LIST OF TRANSFORMATION RULES PREVIOUSLY USED WITHOUT SUCCESS (for reflection):
        {rules}
        --------------------------------------------------
        TRANSFORMATION RULES APPLIED TO TRAIN TASKS AND FEEDBACK FROM THE EVLUATOR:
        {feedback}
        
        --------------------------------------------------
        Now, it's your turn for the next iteration. Use the feedback to improve the prompt and guide the AIs to generate 5 better patterns/rules.
        """

    init_prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        f"""{base_init_prompt}

        Use the "PromptingTool" tool to structure the output correctly based on the description.
        
        Below is the ARC-AGI task that you need to help the AIs to solve:""",
        ),
        ("placeholder", "{messages}"),
    ]
    )
    # LLM setup
    init_llm = setup_llm(
        model_name=model, 
        temperature=temperature, 
        max_tokens=1000, 
        tools=PromptingTool, 
    )
    # chain setup
    init_chain = setup_chain(init_prompt, init_llm)
    
    return init_chain


def node_initiate(state: GraphState, config):
    init_model = config["configurable"]["initiator_model"]

    task_string = state["task_string"] # messages[0][1]
    messages = state["messages"]
    error = state["error"]
    feedback = state["feedback"]
    rules = state["rules"]
    iterations = state["iterations"]
    max_reflections = state["max_reflections"]
    print(f"\n\n------[{iterations+1}/{max_reflections} reflections] INITIATING SYSTEM AND GENERATING PROMPTS------")

    # We have been routed back to generation with an error
    if error == "yes":
        messages += [("user", "Now, try again. Invoke the code tool to structure the output:",)]
        print("Error in the previous step. Try again.")
    
    # langchain.debug = True if feedback is not None else False

    # chain setup
    init_chain = agent_initiation(
                                init_model, 
                                feedback=feedback,
                                rules=rules,
                                temperature=0.1
                                )
    
    # Invoke graph
    init_prompt = init_chain.invoke({"messages": [("user", task_string)]})    
    # langchain.debug = False

    gen_prompt = f"Initial prompt: \n{init_prompt.gen_prompt}"
    print(f"INITIATOR PROMPT: {gen_prompt}\n\n")
    
    return {
        "messages": [("user", gen_prompt)], 
        "gen_prompt": gen_prompt,
        }