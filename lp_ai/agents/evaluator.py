from lp_ai.agents.base import setup_llm, setup_prompt, setup_chain
from lp_ai.output.parsing import check_output, parse_output, insert_errors
from lp_ai.graph.state import GraphState
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# TOOL
# Data model
class EvaluatePatternsTool(BaseModel):
    """
    The evaluator should look at the patterns identified by the combinator and 
    apply them to each example AND test case and EVALUATE the output with a score.
    """
    examples_with_patterns: str = Field(description="Try the patterns provided by the combinator to the challenge examples and test. Show the results")
    evaluation: str = Field(description="Describe and evaluate each of the predictions (using the patterns) with the real examples outputs.")
    score: int = Field(description="Evaluate the patterns and the solutions for the challenges (only one int from 0 to 10).")
    description = "Schema for evaluating the patterns applied to the challenge."


def agent_evaluate(ai_answers, temperature=0.0):

    evaluator_prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        """You are a VERY SMART AI called {llm_name} who is very good at solving puzzles. Below is a list of input and output pairs with a pattern.
        Your job is to look at the problem and compare the responses from the AIs (patterns already identified) and select the best patterns, analyze them and combine them. Also use your own knowledge about the problem.
        Hint: imagine the problem as a grid. Each number represents a different color. Imagine it visually and identify the pattern. Be very careful with the shape of the grids and identify the patterns for the inputs and outputs.
        Your GOAL is to apply the patterns and get the output for the test example. 
        INVOKE THE "EvaluatePatternsTool" tool to structure the output correctly.
        """+"\n"+f"{ai_answers}",
        ),
        ("placeholder", "{messages}"),
    ]
    )   
    # LLM setup
    evaluator_llm = setup_llm(
        model_name="llama3.1", 
        temperature=temperature, 
        max_tokens=1000, 
        tools=EvaluatePatternsTool, 
    )
    # chain setup
    evaluator_chain = setup_chain(evaluator_prompt, evaluator_llm, retries=3)
    
    return evaluator_chain


def node_evaluate_patterns(state: GraphState):
    print("---APPLYING AND EVALUATING PATTERNS---")

    messages = state["messages"]
    error = state["error"]
    iterations = state["iterations"]
    
    print(f'EVALUATOR Messages:')
    task_string = messages[0][1]
    ai_answers = ""
    for message in messages[1:]:
        ai_answers += f"-------------------------------------------------------------\n{message[1]}"
    print(ai_answers)

    # We have been routed back to generation with an error
    if error == "yes":
        messages += [("user", "Now, try again. Invoke the code tool to structure the output:",)]

    # chain setup
    evaluator_chain = agent_evaluate(ai_answers)
    
    # Invoke graph
    evaluation = evaluator_chain.invoke(
        {"llm_name": "llama3.1", 
        "messages": [("user", task_string)]},
    )
    # Increment
    iterations = iterations + 1

    # Print evaluations
    print(f"\n\nEXAMPLES WITH PATTERNS:")
    print(f"{evaluation.examples_with_patterns}")
    print(f"\nEVALUATION: \n{evaluation.evaluation}")
    print(f"\nSCORE: {evaluation.score}")

    return {"evaluation": evaluation.evaluation, 
            "score": evaluation.score,
            "messages": [("assistant", evaluation)],
            "error": state['error'],
            "iterations": iterations
            }