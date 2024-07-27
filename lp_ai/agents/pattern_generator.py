import random
from lp_ai.agents.base import setup_llm, setup_prompt
from lp_ai.graph.state import GraphState

def generate_patterns(state: GraphState):
    messages = state["messages"]
    error = state["error"]

    if error == "yes":
        messages += [("user", "Now, try again. Invoke the code tool to structure the output with a reasoning, imports, and code block:",)]

    temperature = random.uniform(0, 1)
    gen_llm = setup_llm("llama3.1", temperature)

    gen_prompt = setup_prompt(
        """You are a VERY SMART AI called {llm_name} who is very good at solving puzzles. Below is a list of input and output pairs with a pattern.
        First say your name. You need to identify 5 patterns that fit the training examples and enumerate them. ONLY LIST THE PATTERNS.
        Hint: imagine the problem as a grid. Each number represents a different color. Imagine it visually and identify the pattern. Be very careful with the shape of the grids and identify the patterns for the inputs and outputs.""", 
        # messages
    )
    gen_chain = (gen_prompt | gen_llm)
    print(f"---GENERATING PATTERNS llama3.1_{temperature}---")
    patterns = gen_chain.invoke({"llm_name": f"llama3.1_{temperature}", "messages": messages})

    return {"messages": [("assistant", patterns.content)]}