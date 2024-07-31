from langgraph.constants import Send
from langgraph.graph import StateGraph, END, START
from lp_ai.graph.state import GraphState
from lp_ai.agents.pattern_generator import node_generate_patterns
from lp_ai.agents.combinator import node_combine_patterns
from lp_ai.agents.evaluator import node_evaluate_patterns

max_iterations = 3

def decide_to_finish(state: GraphState):
    error = state["error"]
    if error == "no":
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "combinator"
    
def retry_generator(state: GraphState):
    # print(f"RETRY_GENERATOR: {state}")
    error = state["error"]
    iterations = state["iterations"]
    if (error == "no" or error is None) or iterations == max_iterations:
        return "combinator"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "generator"

def evaluation_good_enough(state: GraphState):
    error = state["error"]
    score = state["score"]
    iterations = state["iterations"]
    if ((error == "no" or error is None) and score >= 7) or iterations == max_iterations:
        print(f"\n\n---DECISION: FINISH (Score {score} Good enough)---")
        return "end"
    else:
        print(f"\n\n---DECISION: RETHINK (Score {score} NOT Good enough)---")
        return "combinator"

def setup_workflow(num_generators=3):
    workflow = StateGraph(GraphState)

    # Define the nodes
    for i in range(num_generators):
        workflow.add_node(f"generator_{i}", node_generate_patterns)
    workflow.add_node("combinator", node_combine_patterns)
    workflow.add_node("evaluator", node_evaluate_patterns)

    # Build graph
    for i in range(num_generators):
        workflow.add_edge(START, f"generator_{i}")
        workflow.add_edge(f"generator_{i}", "combinator")
    workflow.add_edge("combinator", "evaluator")
    workflow.add_conditional_edges(
        "evaluator",
        evaluation_good_enough,
        {
            "end": END,
            "combinator": "combinator",
        },
    )
    app = workflow.compile()
    return app