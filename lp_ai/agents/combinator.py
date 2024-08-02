from lp_ai.agents.base import setup_llm, setup_prompt, setup_chain
from lp_ai.output.parsing import check_output, parse_output, insert_errors
from lp_ai.graph.state import GraphState
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


# TOOL
# Data model
class CombinePatternsTool(BaseModel):
    patterns: str = Field(description="Enumerate the patterns/rules combined and used to solve the problem.")
    test_output: str = Field(description="Output for the TEST case (applying the rules).")
    description = "Schema for transformation rules combined in the challenge's task. Don't use brackets {} in the responses."


def agent_combine_patterns(ai_answers, model, temperature=0.0):
    print(f'COMBINATOR MODEL: {model}')

    combinator_prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        """You are a VERY SMART AI called {llm_name} who is very good at solving puzzles. 
        We're solving ARC-AGI, a set of puzzles that require understanding patterns and rules to transform input grids into output grids.
        Each ARC-AGI task consists of a set of input-output examples that follow a pattern (a rule of transformation from input to output).
        THE GOAL OF THE WHOLE SYSTEM IS TO DISCOVER WHAT THOSE TRANSFORMATION RULES ARE (they should work for each example in the task).

        Some other AIs have already generated some patterns/rules for the task. They might be correct or not.
        Your job is to look at the problem and compare the responses from the AIs (patterns/rules already identified) and select the best patterns/rules, analyze them and combine them. Also use your own knowledge about the problem.
        Hint: imagine the problem as a grid. Each number represents a different color. Imagine it visually and identify the pattern. Be very careful with the shape of the grids and identify the patterns/rules for the inputs and outputs.
        
        INVOKE THE "CombinePatternsTool" tool to structure the output correctly. You should:
        1. Take each of the patterns/rules generated by the AIs and apply them to each training example input to get the transformed training outputs.
        2. Test if the generated training outputs are the same as the real training outputs. If not, discard the transformation rule.
        3. Select and list the best transformation rules that work for all the training examples.
        4. Apply the selected transformation rules to the test input example and generate the test output.
        
        
        """+"\nTRANSFORMATION RULES GENERATED BY OTHER AIS:\n"+f"{ai_answers}",
        ),
        ("placeholder", "{messages}"),
    ]
    )
    # LLM setup
    combinator_llm = setup_llm(
        model_name=model,
        temperature=temperature, 
        max_tokens=1000, 
        tools=CombinePatternsTool, 
    )
    # chain setup
    combinator_chain = setup_chain(combinator_prompt, combinator_llm)
    
    return combinator_chain


def node_combine_patterns(state: GraphState, config):
    combinator_model = config["configurable"]["combinator_model"]

    task_string = state["task_string"] # messages[0][1]
    n_generators = state["n_generators"]
    messages = state["messages"]
    error = state['error']
    iterations = state["iterations"]
    max_reflections = state["max_reflections"]
    print(f"\n\n\n------[{iterations+1}/{max_reflections} reflections] COMBINING PATTERNS AND GENERATING SOLUTION------")
    
    print(f'COMBINATOR Messages:')
    ai_answers = ""
    for message in messages[-n_generators:]: # take the last n_generators messages
        ai_answers += f"\n-------------------------------------------------------------\n{message[1]}"
    print(ai_answers)

    # chain setup
    combinator_chain = agent_combine_patterns(ai_answers, combinator_model, 0.1)
    
    # Invoke graph
    patterns_combined = combinator_chain.invoke(
        {"llm_name": "COMBINATOR_"+combinator_model,
        "messages": [("user", task_string)]},
    )
    return {"test_output": patterns_combined.test_output, 
            "messages": [("assistant", patterns_combined)],
            "error": error,
            "rules": patterns_combined.patterns,
            }