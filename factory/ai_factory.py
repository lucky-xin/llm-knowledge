import os

from langchain_community.llms.tongyi import Tongyi
from langchain_core.language_models import BaseLLM
from langchain_openai import ChatOpenAI
from llama_index.llms.openai_like import OpenAILike


def create_chat_ai() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen2.5-14b-instruct-1m",
        # model="qwen-turbo-latest",
        temperature=0.2
    )


def create_glm_chat_ai() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        base_url="https://open.bigmodel.cn/api/paas/v4/",
        model="glm-4",
        # model="qwen-turbo-latest",
        temperature=0.2
    )


def create_llm() -> BaseLLM:
    return Tongyi(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen2.5-14b-instruct-1m",
    )


from llama_index.core.llms.llm import LLM


def create_llama_index_llm() -> LLM:
    return OpenAILike(
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-turbo-latest",
        is_chat_model=True,
        context_window=30000
    )
