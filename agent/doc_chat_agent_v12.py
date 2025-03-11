import threading
import uuid
from typing import Optional, List, Sequence

from langchain_community.document_loaders import WikipediaLoader
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import GraphCypherQAChain
from langchain_text_splitters import TokenTextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from llama_index.core import SimpleDirectoryReader, Document, Settings
from llama_index.core.ingestion import run_transformations
from llama_index.core.schema import BaseNode
from llama_index.core.vector_stores import VectorStoreQuery, MetadataFilter, FilterOperator, MetadataFilters

from adapter import LangchainDocumentAdapter, LLamIndexDocumentAdapter
from entities import State
from factory.age_graph import create_age_graph
from factory.llm import LLMFactory, LLMType
from factory.neo4j import create_neo4j_graph
from factory.store_index import create_vector_store_index
from factory.vector_store import create_pg_vector_store
from utils import create_combine_prompt, \
    convert_to_graph_documents

llm_factory = LLMFactory(
    llm_type=LLMType.LLM_TYPE_QWENAI,
)
llm = llm_factory.create_chat_llm()
llm_transformer = LLMGraphTransformer(llm=llm)
vector_store = create_pg_vector_store()
index = create_vector_store_index(vector_store)
age_graph = create_neo4j_graph()
graph_cypher_qa_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=age_graph,
    verbose=True,
    validate_cypher=True,
    use_function_response=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True,
)


def fetch(doc: Document, c: Optional[RunnableConfig] = None):
    return llm_transformer.process_response(doc, c)


def query_by_ids(ids: List[str]) -> Sequence[BaseNode]:
    """批量查询节点数据，返回 Node列表"""
    # 返回原始字典格式
    try:
        result = vector_store.query(
            VectorStoreQuery(
                node_ids=ids,
                query_embedding=[],
                similarity_top_k=3,
                embedding_field="embedding",
                filters=MetadataFilters(
                    filters=[
                        MetadataFilter(
                            key="id",
                            value=ids,
                            operator=FilterOperator.IN
                        ),
                    ]
                ),
            )
        )
        return result.nodes
    except Exception as e:
        print(f"批量查询失败: {e}")
        return []


# # Read the wikipedia article
def add_wiki_docs():
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
    graph_documents = convert_to_graph_documents(langchain_documents, fetch)
    age_graph.add_graph_documents(graph_documents)


def add_docs():
    reader = SimpleDirectoryReader("/tmp/agent")
    documents: list[Document] = reader.load_data()
    nodes = run_transformations(
        nodes=documents,
        transformations=Settings.transformations
    )
    langchain_document_adapter = LangchainDocumentAdapter()
    graph_documents = convert_to_graph_documents(langchain_document_adapter(nodes), fetch)
    age_graph.add_graph_documents(graph_documents)
    index.insert_nodes(nodes)


add_wiki_docs()
add_docs()


def graph_retriever_chain(state: State):
    messages = state.get("messages")
    question = messages[-1].content
    resp = graph_cypher_qa_chain.invoke({"query": question})
    graph_data = resp.get("result")
    return {"graph_data": graph_data, "question": question}


def vector_store_retriever_chain(state: State):
    messages = state.get("messages")
    question = messages[-1].content
    embedding = Settings.embed_model.get_text_embedding(question)

    resp = index.vector_store.query(
        VectorStoreQuery(
            query_embedding=embedding,
            query_str=question
        )
    )
    vector_data = [el.get_content() for el in resp.nodes]
    return {"vector_data": "\n".join(vector_data)}


query_engine = index.as_query_engine()
print("Creating graph...")
# 初始化 MemorySaver 共例
workflow = StateGraph(State)


def sender_chain_v2(state: State):
    graph_data = graph_retriever_chain(state)
    vector_data = vector_store_retriever_chain(state)
    resp = dict(graph_data, **vector_data)
    return resp


def searcher_chain(state: State):
    chain = create_combine_prompt() | llm
    resp = chain.invoke(state)
    return {"messages": [resp]}


def should_continue_v1(state: State):
    vector_data = state.get("vector_data")
    graph_data = state.get("graph_data")
    answer = state.get("answer")
    if answer:
        return END
    if vector_data and graph_data:
        return "searcher"
    if not vector_data and graph_data:
        return "vector_store_retriever"
    if not graph_data and vector_data:
        return "graph_retriever"
    return "sender"


def should_continue_v2(state: State):
    messages = state.get("messages")
    if not messages and not isinstance(messages[-1], HumanMessage):
        return END
    vector_data = state.get("vector_data")
    graph_data = state.get("graph_data")
    answer = state.get("answer")
    if answer:
        return END
    if vector_data and graph_data:
        return "searcher"
    if not vector_data and graph_data:
        return "vector_store_retriever"
    if not graph_data and vector_data:
        return "graph_retriever"
    return "sender"


workflow.add_edge(START, "sender")
workflow.add_node("sender", sender_chain_v2)
workflow.add_node("searcher", searcher_chain)

workflow.add_edge("sender", "searcher")
workflow.add_edge("searcher", END)

workflow.add_conditional_edges("sender", should_continue_v2)
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
run_id = str(uuid.uuid4())

config = {
    "recursion_limit": 50,
    "configurable": {
        "run_id": run_id,
        "thread_id": str(threading.current_thread().ident)
    },
    # "callbacks": [st_cb]
}
while True:
    q = input("请问我任何关于文章的问题")
    if q:
        stream = graph.stream(
            input={"messages": [HumanMessage(content=q)]},
            config=config,
            stream_mode="messages"
        )
        collected_messages = ""
        for chunks in stream:
            for chunk in chunks:
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    collected_messages += chunk.content
                    print(collected_messages + "▌")
