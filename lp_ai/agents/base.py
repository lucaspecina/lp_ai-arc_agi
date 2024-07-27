from langchain_ollama import ChatOllama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import os 

# Base agent class or functions can be defined here
# E.g., LLM setup, common utilities, etc.

# Example of a base LLM setup
def setup_llm(model_name, temperature=0, max_tokens=1000):
    if model_name == 'llama3.1':
        return ChatOllama(model=model_name, temperature=temperature, max_tokens=max_tokens)
    elif model_name == 'gpt-4o':
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OpenAI API key is not defined.")
        return ChatOpenAI(model=model_name, openai_api_key=openai_api_key, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

# Example of a base prompt setup
def setup_prompt(template_string):
    return ChatPromptTemplate.from_messages(
        [
            ("system", template_string),
            ("placeholder", "{messages}"),
        ]
    )
