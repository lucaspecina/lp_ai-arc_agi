from lp_ai.agents.base import setup_llm, setup_prompt, setup_chain
from lp_ai.output.parsing import check_output, parse_output, insert_errors
from lp_ai.graph.state import GraphState
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from typing import List
from langchain_core.tools import tool

# TOOLS

# TODO: evaluator shouldn't be able to see the training examples real output if it generates the training outputs applying the rules (it might copy them directly)

# Data model
class EvaluatePatternsTool(BaseModel):
    training_outputs_with_rules: List[str] = Field(description="Generated outputs for the training examples, applying the rules (each training example is one element (string) in the list).")
    score: int = Field(description="Evaluate the patterns/rules and the solutions for the challenges (only one int from 0 to 10).")
    feedback: str = Field(description="Feedback for the combinator on the evaluation. Recommend improvements.")
    description = "Schema for evaluating the patterns applied to the challenge. Don't use brackets {} in the responses."



def agent_evaluate(combinator_solution, model, temperature=0.0):
    print(f'EVALUATOR MODEL: {model}')

    evaluator_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a VERY SMART AI called {llm_name} who excels at evaluating patterns. 
                We're solving ARC-AGI, a set of puzzles that require understanding patterns and rules to transform input grids into output grids.
                Each ARC-AGI task consists of a set of input-output examples that follow a pattern (a rule of transformation from input to output).
                THE GOAL OF THE WHOLE SYSTEM IS TO DISCOVER WHAT THOSE TRANSFORMATION RULES ARE (they should work for each example in the task).

                Another AI, called the Combinator, already proposed some transformation rules for this particular task.
                
                INVOKE THE "EvaluatePatternsTool" tool to structure the output correctly. You should:
                1. Take each of the patterns/rules proposed by the Combinator and apply them to each training example input to get the transformed training outputs (and save them in the right format).
                2. Test if the generated training outputs are the same as the real training outputs.
                3. Based on the results, evaluate the patterns/rules and the solutions for the challenges.
                4. Give a score from 0 to 10 based on the accuracy of the patterns/rules and the solution for the test case.
                5. Provide extensive feedback to the combinator on the evaluation. Explain what's wrong and what to consider next.
                Be VERY STRICT and Don't score more than 7 if applying the patterns/rules doesn't produce the exact real outputs.
                
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
    combinator_solution += f"Combinator solution: \n\nTransformation rules:\n{patterns_combined}\n\nTest output:\n{test_output}"
    print(combinator_solution)

    # chain setup
    evaluator_chain = agent_evaluate(combinator_solution, evaluator_model, 0.1)
    
    # Invoke graph
    evaluation = evaluator_chain.invoke(
        {"llm_name": "EVALUATOR"+evaluator_model,
        "messages": [("user", task_string)]},
    )

    # Print evaluations
    feedback_message = f"EVALUATION OF TRAIN EXAMPLES WITH RULES APPLIED:\n{evaluation.training_outputs_with_rules}\n\nFEEDBACK:\n{evaluation.feedback}\n"
    print(f"\n\n{feedback_message}")
    print(f"\nSCORE: {evaluation.score}")

    return {"feedback": feedback_message,
            "score": evaluation.score,
            "messages": [("assistant", evaluation)],
            "error": error,
            "iterations": iterations + 1,
            "training_predictions": evaluation.training_outputs_with_rules,
            }