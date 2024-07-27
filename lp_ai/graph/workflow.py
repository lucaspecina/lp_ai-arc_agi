from langgraph.graph import StateGraph, END, START
from lp_ai.graph.state import GraphState
from lp_ai.agents.pattern_generator import generate_patterns
from lp_ai.agents.combinator import combination


def decide_to_finish(state: GraphState):
    error = state["error"]
    if error == "no":
        print("---DECISION: FINISH---")
        return "end"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "combinator"

def setup_workflow(num_generators=3):
    workflow = StateGraph(GraphState)

    # Define the nodes
    for i in range(num_generators):
        workflow.add_node(f"generator_{i}", generate_patterns)
    workflow.add_node("combinator", combination)

    # Build graph
    for i in range(num_generators):
        workflow.add_edge(START, f"generator_{i}")
        workflow.add_edge(f"generator_{i}", "combinator")

    workflow.add_edge("combinator", END)
    app = workflow.compile()
    return app