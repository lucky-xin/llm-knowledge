import json
import threading
import uuid
from typing import Optional, Dict, Any, List

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import GraphCypherQAChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from llama_index.core import SimpleDirectoryReader, Document, Settings
from llama_index.core.ingestion import run_transformations
from llama_index.core.vector_stores import VectorStoreQuery
from pyvis.network import Network

from adapter import LangchainDocumentAdapter, LLamIndexDocumentAdapter
from entities import State
from factory.llm import LLMFactory, LLMType
from factory.neo4j import create_neo4j_graph
from factory.store_index import create_vector_store_index
from factory.vector_store import create_neo4j_vector_store
from utils import create_combine_prompt, convert_to_graph_documents

llm_factory = LLMFactory(
    llm_type=LLMType.LLM_TYPE_QWENAI,
)
llm = llm_factory.create_llm()
chat_llm = llm_factory.create_chat_llm()
llm_transformer = LLMGraphTransformer(llm=llm)
vector_store = create_neo4j_vector_store()
index = create_vector_store_index(vector_store)
neo4j_graph = create_neo4j_graph()
graph_cypher_qa_chain = GraphCypherQAChain.from_llm(
    llm=llm,
    input_key="question",
    top_k=10,
    verbose=True,
    return_direct=False,
    validate_cypher=True,
    use_function_response=False,
    allow_dangerous_requests=True,
    return_intermediate_steps=True,
    graph=neo4j_graph,
)
def fetch(doc: Document, c: Optional[RunnableConfig] = None):
    return llm_transformer.process_response(doc, c)

# # Read the wikipedia article
def add_wiki_docs():
    loader = WebBaseLoader(
        "https://zh.wikipedia.org/wiki/%E5%93%AA%E5%90%92%E4%B9%8B%E9%AD%94%E7%AB%A5%E9%97%B9%E6%B5%B7")
    raw_documents = loader.load()
    # # Define chunking strategy
    langchain_documents = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    ).split_documents(raw_documents)

    llam_index_document_adapter = LLamIndexDocumentAdapter()
    llam_index_documents = llam_index_document_adapter(langchain_documents)

    nodes = run_transformations(
        nodes=llam_index_documents,
        transformations=Settings.transformations
    )
    index.insert_nodes(nodes)
    graph_documents = convert_to_graph_documents(langchain_documents, fetch)
    neo4j_graph.add_graph_documents(graph_documents)


def add_docs():
    reader = SimpleDirectoryReader("/tmp/agent")
    documents: list[Document] = reader.load_data()
    nodes = run_transformations(
        nodes=documents,
        transformations=Settings.transformations
    )
    langchain_document_adapter = LangchainDocumentAdapter()
    graph_documents = convert_to_graph_documents(langchain_document_adapter(nodes), fetch)
    neo4j_graph.add_graph_documents(graph_documents)
    index.insert_nodes(nodes)


# add_wiki_docs()
# add_docs()


def graph_retriever_chain(state: State):
    messages = state.get("messages")
    question = messages[-1].content
    resp = graph_cypher_qa_chain.invoke({"question": question})
    graph_data = resp.get("result")
    steps = resp.get("intermediate_steps", [])
    cypher = steps[0].get("query", '') if steps else ''
    return {
        "graph_data": json.dumps(graph_data, indent=2, ensure_ascii=False),
        "question": question,
        "cypher": cypher
    }


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


def sender_chain(state: State):
    graph_data = graph_retriever_chain(state)
    vector_data = vector_store_retriever_chain(state)
    resp = dict(graph_data, **vector_data)
    return resp


def searcher_chain(state: State):
    chain = create_combine_prompt() | chat_llm
    resp = chain.invoke(state)
    return {"messages": [resp]}


def should_continue(state: State):
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


# 可视化生成函数
def generate_visualization(data: List[Dict[str, Any]]):
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#1E1E1E",
        font_color="white",
        directed=True
    )

    # 设置布局参数
    net.barnes_hut()

    for datum in data:
        # 添加节点和边
        for node in datum["nodes"]:
            net.add_node(
                n_id=node["id"],
                label=node["label"],
                title=json.dumps(node["properties"], indent=2),
                color="#4CAF50" if node["type"] == "Person" else "#2196F3",
                shape="dot" if node["type"] == "Entity" else "diamond"
            )

        for edge in datum["edges"]:
            net.add_edge(
                source=edge["source"],
                to=edge["target"],
                label=edge["type"],
                color="#FF9800",
                width=2
            )

    # 生成HTML文件
    net.save_graph("temp.html")
    return open("temp.html", "r", encoding="utf-8").read()

workflow.add_edge(START, "sender")
workflow.add_node("sender", sender_chain)
workflow.add_node("searcher", searcher_chain)

workflow.add_edge("sender", "searcher")
workflow.add_edge("searcher", END)

workflow.add_conditional_edges("sender", should_continue)



# graph = workflow.compile(checkpointer=create_checkpointer(), store=create_store())
graph = workflow.compile(checkpointer=InMemorySaver())
run_id = str(uuid.uuid4())

config = {
    "recursion_limit": 50,
    "configurable": {
        "run_id": run_id,
        "thread_id": str(threading.current_thread().ident)
    },
    # "callbacks": [st_cb]
}
cypher = "MATCH (p:Person{id:\"哪吒\"})-[r]-(c) RETURN p, r, c"
values = neo4j_graph.query(cypher)
html = generate_visualization(values)
print(html)

while True:
    q = input("请问我任何关于文章的问题")
    if q:
        collected_messages = ""
        stream = graph.stream(
            input={"messages": [HumanMessage(content=q)]},
            config=config,
            stream_mode="messages"
        )
        for chunks in stream:
            for chunk in chunks:
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    collected_messages += chunk.content
            print(collected_messages + "\n---")
