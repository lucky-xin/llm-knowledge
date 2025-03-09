from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.embeddings.dashscope import DashScopeTextEmbeddingModels, DashScopeEmbedding
from pydantic import BaseModel, Field

from factory.ai_factory import create_openai_chat_ai
from langchain_experimental.graph_transformers import LLMGraphTransformer

class Vertex(BaseModel):
    sur: str = Field(description="起始节点名称")
    dst: str = Field(description="结束节点名称")
    relation: str = Field(description="关系名称")


class Vertexes(BaseModel):
    vertexes: List[Vertex] = Field(description="知识图谱节点列表")


class KnowledgeProcessor:
    def __init__(self):
        self.embedder: BaseEmbedding = DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3)
        self.llm: BaseChatModel = create_openai_chat_ai()
        self.triple_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    从文本中提取实体和关系：
                    文本：{text}
                    严格按照JSON数组格式进行输出，JSON数组每个元素字段如下：
                    sur: 起始节点名称
                    dst: 结束节点名称
                    relation: 关系名称
                    """
                )
            ]
        )

    def extract_triples(self, text: str) -> List[Vertex]:
        chain = self.triple_prompt | self.llm.with_structured_output(schema=Vertexes, method="function_calling")
        result: Vertexes = chain.invoke(input={"text": text})
        return result.vertexes

    def embed_text(self, text: str):
        return self.embedder.embed_documents([text])[0]
