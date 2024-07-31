# ARC-AGI-solution
 
Trying out different ways of “thinking”, using LLMs as my intuition guide for the search process. The goal is to solve ARC-AGI but we'll see.

## Current version

The code is still in the early stages but the idea is to have a simple and fast way to test different approaches and see what works. 

The code is not optimized for performance, it's optimized for flexibility and ease of use.
So far it's as simple as:
- multiple "intuition" pattern generators (parallel LLMs identifying patterns in the task)
- combinator that takes those patterns, combine them and generate a solution for the test
- evaluator that checks if the solution is correct, scores the output and gives feedback to the generators 

## Requirements:
- set and .env with your OPENAI_API_KEY
- install ollama and get llama3.1

## Usage
```bash
python main.py  --num_generators 10 --num_iterations 1 --combinator_model gpt-4o --evaluator_model llama3.1 --task_id 0520fde7
```

## Contributing

Ideas and pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.