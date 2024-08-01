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
    examples_with_patterns: str = Field(description="Apply the patterns/rules provided by the combinator to the challenge examples and test if they're the same as the real example's outputs. Don't include the inputs.")
    score: int = Field(description="Evaluate the patterns/rules and the solutions for the challenges (only one int from 0 to 10).")
    feedback: str = Field(description="Feedback for the combinator on the evaluation. Recommend improvements.")
    description = "Schema for evaluating the patterns applied to the challenge."


def agent_evaluate(combinator_solution, model, temperature=0.0):
    print(f'EVALUATOR MODEL: {model}')

    evaluator_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a VERY SMART AI called {llm_name} who excels at evaluating patterns. We're trying to solve a puzzle that consist on a set of input and output pairs with a pattern/rule.
                Below is a list of patterns/rules identified by the combinator.
                
                Your task is to apply these patterns/rules to each example and test case, evaluate the results, and score them.
                Carefully analyze the application of each pattern/rules and compare the predicted outputs with the actual outputs.
                Your GOAL is to provide a detailed evaluation of the predictions and assign a score based on the accuracy of the patterns/rules.
                
                Be VERY STRICT and Don't score more than 7 if applying the patterns/rules doesn't produce the exact real outputs.
                Provide extensive feedback to the combinator on the evaluation. Explain what's wrong and what to consider next.
                
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
        max_tokens=1000, 
        tools=EvaluatePatternsTool, 
    )
    # chain setup
    evaluator_chain = setup_chain(evaluator_prompt, evaluator_llm)
    
    return evaluator_chain


def node_evaluate_patterns(state: GraphState, config):
    evaluator_model = config["configurable"]["evaluator_model"]

    task_string = state["task_string"] # messages[0][1]
    messages = state["messages"]
    error = state["error"]
    iterations = state["iterations"]
    max_reflections = state["max_reflections"]
    patterns_combined = state["rules"]
    test_output = state["test_output"]
    print(f"\n\n------[{iterations+1}/{max_reflections} reflections] EVALUATING AND APPLYING PATTERNS------")
    
    print(f'EVALUATOR Messages:')
    
    combinator_solution = "-------------------------------------------------------------\n"
    combinator_solution += f"Combinator solution: \n\nPatterns:\n{patterns_combined}\n\nTest output:\n{test_output}"
    print(combinator_solution)

    # chain setup
    evaluator_chain = agent_evaluate(combinator_solution, evaluator_model, 0.3)
    
    # Invoke graph
    evaluation = evaluator_chain.invoke(
        {"llm_name": "EVALUATOR"+evaluator_model,
        "messages": [("user", task_string)]},
    )

    # Print evaluations
    feedback_message = f"EVALUATION OF TRAIN EXAMPLES WITH RULES APPLIED:\n{evaluation.examples_with_patterns}\n\nFEEDBACK:\n{evaluation.feedback}\n"
    print(f"\n\n{feedback_message}")
    print(f"\nSCORE: {evaluation.score}")

    return {"feedback": feedback_message,
            "score": evaluation.score,
            "messages": [("assistant", evaluation)],
            "error": error,
            "iterations": iterations + 1
            }