import os
from dataclasses import dataclass
from enum import Enum

from langchain_community.chat_models import ChatTongyi, ChatZhipuAI
from langchain_community.llms.tongyi import Tongyi
from langchain_core.language_models import BaseLLM, BaseChatModel
from langchain_openai import ChatOpenAI, OpenAI
from llama_index.embeddings.dashscope import DashScopeTextEmbeddingModels, DashScopeEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike


class LLMType(str, Enum):
    """DashScope TextEmbedding models."""

    LLM_TYPE_QWENAI = "qwen"
    LLM_TYPE_OPENAI = "openai"
    LLM_TYPE_ZHIPUAI = "zhipu"


@dataclass
class LLMFactory:
    llm_type: LLMType = LLMType.LLM_TYPE_QWENAI
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key: str = os.getenv("DASHSCOPE_API_KEY")
    model: str = "qwq-plus-latest"
    temperature: float = 0.51

    def create_chat_llm(self) -> BaseChatModel:
        if self.llm_type == LLMType.LLM_TYPE_OPENAI:
            return ChatOpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                model=self.model,
                temperature=self.temperature
            )
        elif self.llm_type == LLMType.LLM_TYPE_QWENAI:
            return ChatTongyi(
                api_key=self.api_key,
                model=self.model
            )
        elif self.llm_type == LLMType.LLM_TYPE_ZHIPUAI:
            return ChatZhipuAI(
                api_key=self.api_key,
                api_base=self.api_base,
                model=self.model,
                temperature=self.temperature
            )
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")

    def create_llm(self) -> BaseLLM:
        if self.llm_type == LLMType.LLM_TYPE_OPENAI:
            return OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                model=self.model,
                # model="qwen-turbo-latest",
                temperature=0.2
            )
        elif self.llm_type == LLMType.LLM_TYPE_QWENAI:
            return Tongyi(
                api_key=self.api_key,
                model=self.model,
            )
        elif self.llm_type == LLMType.LLM_TYPE_ZHIPUAI:
            return OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                model=self.model,
                # model="qwen-turbo-latest",
                temperature=0.2
            )
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")

    from llama_index.core.llms.llm import LLM

    def create_llama_index_llm(self) -> LLM:
        return OpenAILike(
            api_base=self.api_base,
            api_key=self.api_key,
            model=self.model,
            is_chat_model=True,
            context_window=30000
        )

    def llama_index_embedding(self):
        if self.llm_type == LLMType.LLM_TYPE_OPENAI:
            return OpenAIEmbedding(
                model=self.model,
            )
        elif self.llm_type == LLMType.LLM_TYPE_QWENAI:
            return DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3)
        elif self.llm_type == LLMType.LLM_TYPE_ZHIPUAI:
            return OpenAIEmbedding(
                model=self.model,
                api_base="https://open.bigmodel.cn/api/paas/v4/embeddings",
            )
        else:
            raise ValueError(f"Unsupported LLM type: {self.llm_type}")
