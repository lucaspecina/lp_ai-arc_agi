from langgraph.constants import Send
from langgraph.graph import StateGraph, END, START
from lp_ai.graph.state import GraphState
from lp_ai.agents.pattern_generator import node_generate_patterns
from lp_ai.agents.combinator import combination
from lp_ai.agents.pattern_generator import agent_generate_patterns


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
    if error == "no" or error is None:
        return "combinator"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "generator"
    

# def retry_generator(state: GraphState):
#     print(f"RETRY_GENERATOR: {state}")
#     error = state["error"]
#     if error == "no":
#         return "combinator"
#     else:
#         print("---DECISION: RE-TRY SOLUTION---")
#         return [Send("generator", {"subject": s}) for s in state['errors']]


def setup_workflow(num_generators=3):
    workflow = StateGraph(GraphState)

    # Define the nodes
    for i in range(num_generators):
        workflow.add_node(f"generator_{i}", node_generate_patterns)
    workflow.add_node("combinator", combination)

    # Build graph
    for i in range(num_generators):
        workflow.add_edge(START, f"generator_{i}")
        # workflow.add_edge(f"generator_{i}", "combinator")
        workflow.add_conditional_edges(
            f"generator_{i}",
            retry_generator,
            {
                "combinator": "combinator",
                "generator": f"generator_{i}",
            },
    )
    workflow.add_edge("combinator", END)
    app = workflow.compile()
    return app