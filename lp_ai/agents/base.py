from langchain_ollama import ChatOllama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
import os 
from lp_ai.output.parsing import check_output, parse_output, insert_errors

# Example of a base LLM setup
def setup_llm(model_name, temperature=0, max_tokens=1000, tools=None):
    if model_name.startswith('llama3'):
        llm = ChatOllama(model=model_name, temperature=temperature, max_tokens=max_tokens)
        llm = llm.with_structured_output(tools, include_raw=True)
        return llm
    
    elif model_name.startswith("gpt"):
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OpenAI API key is not defined.")
        llm = ChatOpenAI(model=model_name, openai_api_key=openai_api_key, max_tokens=max_tokens)
        llm = llm.with_structured_output(tools, include_raw=True)
        return llm
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def setup_chain(prompt, llm, retries=5):
    gen_chain_raw = prompt | llm | check_output
    
    # This will be run as a fallback chain
    fallback_chain = insert_errors | gen_chain_raw
    gen_chain_retry = gen_chain_raw.with_fallbacks(fallbacks=[fallback_chain] * retries, exception_key="error")

    return gen_chain_retry | parse_output

# Example of a base prompt setup
def setup_prompt(template_string):
    return ChatPromptTemplate.from_messages(
        [
            ("system", template_string),
            ("placeholder", "{messages}"),
        ]
    )
