import os

from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI, OpenAI


def create_chat_ai() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen2.5-14b-instruct-1m",
        # model="qwen-turbo-latest",
        temperature=0.2
    )


def create_llm() -> BaseLLM:
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen2.5-14b-instruct-1m",
        # model="qwen-turbo-latest",
        temperature=0.2
    )
