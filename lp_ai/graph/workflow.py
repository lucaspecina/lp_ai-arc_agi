from langgraph.constants import Send
from langgraph.graph import StateGraph, END, START
from lp_ai.graph.state import GraphState
from lp_ai.agents.pattern_generator import node_generate_patterns
from lp_ai.agents.combinator import node_combine_patterns
from lp_ai.agents.evaluator import node_evaluate_patterns
from lp_ai.agents.initiator import node_initiate
from lp_ai.output.scoring import test_training_examples
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


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
    max_reflections = state["max_reflections"]
    if (error == "no" or error is None) or iterations == max_reflections:
        return "combinator"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "generator"

def evaluation_good_enough(state: GraphState):
    error = state["error"]
    score = state["score"]
    iterations = state["iterations"]
    max_reflections = state["max_reflections"]
    task_id = state["task_id"]
    training_predictions = state["training_predictions"]

    # TODO: Test the training examples with the rules applied
    test_training_examples(training_predictions, task_id)

    if ((error == "no" or error is None) and score > 8) or iterations == max_reflections:
        print(f"\n\n---DECISION: FINISH (Score {score} Good enough)---")
        return "end"
    else:
        print(f"\n\n---DECISION: RETHINK (Score {score} NOT Good enough)---")
        return "initiator"

def setup_workflow(num_generators=3):
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("initiator", node_initiate)
    for i in range(num_generators):
        workflow.add_node(f"generator_{i}", node_generate_patterns)
    workflow.add_node("combinator", node_combine_patterns)
    workflow.add_node("evaluator", node_evaluate_patterns)

    # Build graph
    workflow.add_edge(START, "initiator")
    for i in range(num_generators):
        workflow.add_edge("initiator", f"generator_{i}")
        workflow.add_edge(f"generator_{i}", "combinator")
    workflow.add_edge("combinator", "evaluator")
    workflow.add_conditional_edges(
        "evaluator",
        evaluation_good_enough,
        {
            "end": END,
            "initiator": "initiator",
        },
    )
    app = workflow.compile()

    # Save the graph image to a file
    # graph_image_path = "workflow_graph.png"
    # graph_image = app.get_graph().draw_mermaid_png()
    # with open(graph_image_path, "wb") as f:
    #     f.write(graph_image)
    
    return app