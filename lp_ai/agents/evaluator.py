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
    examples_with_patterns: str = Field(description="Try the patterns provided by the combinator to the challenge examples and test if they're the same as the real example's outputs. Don't include the inputs.")
    evaluation: str = Field(description="Describe and evaluate each of the predictions (using the patterns).")
    score: int = Field(description="Evaluate the patterns and the solutions for the challenges (only one int from 0 to 10).")
    description = "Schema for evaluating the patterns applied to the challenge."


def agent_evaluate(combinator_solution, model, temperature=0.0):
    print(f'EVALUATOR MODEL: {model}')

    evaluator_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a VERY SMART AI called {llm_name} who excels at evaluating patterns. We're trying to solve a puzzle that consist on a set of input and output pairs with a pattern.
                Below is a list of patterns identified by the combinator.
                Your task is to apply these patterns to each example and test case, evaluate the results, and score them.
                Carefully analyze the application of each pattern and compare the predicted outputs with the actual outputs.
                Your GOAL is to provide a detailed evaluation of the predictions and assign a score based on the accuracy of the patterns.
                INVOKE THE "EvaluatePatternsTool" tool to structure the output correctly.
                """+"\n"+f"{combinator_solution}",
            ),
            ("placeholder", "{messages}"),
        ]
    )
    # LLM setup
    evaluator_llm = setup_llm(
        model_name=model,
        temperature=temperature, 
        max_tokens=3000, 
        tools=EvaluatePatternsTool, 
    )
    # chain setup
    evaluator_chain = setup_chain(evaluator_prompt, evaluator_llm, retries=3)
    
    return evaluator_chain


def node_evaluate_patterns(state: GraphState, config):
    print("\n\n------EVALUATING AND APPLYING PATTERNS------")
    evaluator_model = config["configurable"]["evaluator_model"]

    messages = state["messages"]
    error = state["error"]
    iterations = state["iterations"]
    
    print(f'EVALUATOR Messages:')
    task_string = messages[0][1]
    combinator_solution = "-------------------------------------------------------------\n"
    combinator_solution += f"Combinator solution: \nPatterns:\n{messages[-1][1].patterns}\nTest output:\n{messages[-1][1].test_output}"
    print(combinator_solution)

    # # We have been routed back to generation with an error
    # if error == "yes":
    #     messages += [("user", "Now, try again. Invoke the code tool to structure the output:",)]

    # chain setup
    evaluator_chain = agent_evaluate(combinator_solution, evaluator_model, 0.3)
    
    # Invoke graph
    evaluation = evaluator_chain.invoke(
        {"llm_name": "EVALUATOR"+evaluator_model,
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