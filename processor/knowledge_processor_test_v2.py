from typing import Optional, List, Sequence

from langchain_community.document_loaders import WikipediaLoader
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import GraphCypherQAChain
from langchain_text_splitters import TokenTextSplitter
from llama_index.core import Settings
from llama_index.core.ingestion import run_transformations
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import VectorStoreQuery

from adapter import LLamIndexDocumentAdapter
from factory.age_graph import create_age_graph
from factory.llm import LLMType, LLMFactory
from factory.store_index import create_vector_store_index
from factory.vector_store import create_pg_vector_store

llm_factory = LLMFactory(
    llm_type=LLMType.LLM_TYPE_QWENAI,
)
llm = llm_factory.create_chat_llm()

age_graph = create_age_graph()
llm_transformer = LLMGraphTransformer(llm=llm)
vector_store = create_pg_vector_store()
index = create_vector_store_index(vector_store)
base_query_engine = index.as_query_engine(llm=llm)


def fetch(doc: Document, config: Optional[RunnableConfig] = None):
    return llm_transformer.process_response(doc, config)


def batch_query_by_ids(ids: List[str]) -> Sequence[BaseNode]:
    """批量查询节点数据，返回 Node列表"""
    # 返回原始字典格式
    try:
        result = vector_store.query(
            VectorStoreQuery(
                node_ids=ids,
                query_embedding=[],
                similarity_top_k=3,
                embedding_field="embedding",
                query_str="",
            )
        )
        return result.nodes
    except Exception as e:
        print(f"批量查询失败: {e}")
        return []


# # Read the wikipedia article
def add_docs():
    raw_documents = WikipediaLoader(query="Elizabeth I").load()
    # # Define chunking strategy
    text_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
    langchain_documents = text_splitter.split_documents(raw_documents)

    llam_index_document_adapter = LLamIndexDocumentAdapter()
    llam_index_documents = llam_index_document_adapter(langchain_documents)

    nodes = run_transformations(
        nodes=llam_index_documents,
        transformations=Settings.transformations
    )
    index.insert_nodes(nodes)


# node_ids = [node.node_id for node in nodes]
# res = index.vector_store.query(VectorStoreQuery(node_ids=node_ids))
# exist_ids = res.ids
# graph_documents = convert_to_graph_documents(langchain_documents)
# ng.add_graph_documents(graph_documents)
# inserts = [node for node in nodes if node.node_id not in exist_ids]
# if len(inserts):
#     index.insert_nodes(inserts)


chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=age_graph,
    verbose=True,
    validate_cypher=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
)

add_docs()

question = "Which house did Elizabeth I belong to?"
embedding = Settings.embed_model.get_text_embedding(question)

resp = index.vector_store.query(
    VectorStoreQuery(
        query_embedding=embedding,
        query_str=question
    )
)
print(resp)

resp = chain.invoke({"query": question})
print(resp)
