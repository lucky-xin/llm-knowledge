import json
import threading
import uuid
from typing import Optional

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

from adapter import LangchainDocumentAdapter, LLamIndexDocumentAdapter
from entities import State
from factory.llm import LLMFactory, LLMType
from factory.neo4j import create_neo4j_graph
from factory.store_index import create_vector_store_index
from factory.vector_store import create_neo4j_vector_store
from utils import create_combine_prompt, convert_to_graph_documents, generate_visualization

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


def fetch(llm_transformer, doc: Document, c: Optional[RunnableConfig] = None):
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
    graph_documents = convert_to_graph_documents(llm_transformer, langchain_documents, fetch)
    neo4j_graph.add_graph_documents(graph_documents)


def add_docs():
    reader = SimpleDirectoryReader("/tmp/agent")
    documents: list[Document] = reader.load_data()
    nodes = run_transformations(
        nodes=documents,
        transformations=Settings.transformations
    )
    langchain_document_adapter = LangchainDocumentAdapter()
    graph_documents = convert_to_graph_documents(llm_transformer,langchain_document_adapter(nodes), fetch)
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


def create_html():
    cypher = """
    MATCH (n:Person)
    OPTIONAL MATCH (n)-[r]->(m)
    RETURN 
      id(n) AS n_id,
      labels(n) AS n_labels,
      properties(n) AS n_properties,
      id(m) AS m_id,
      labels(m) AS m_labels,
      properties(m) AS m_properties,
      TYPE(r) AS r_type,
      properties(r) AS r_properties
    """
    values = neo4j_graph.query(cypher)
    nodes = []
    edges = []
    for record in values:
        n_id = record.get('n_id', None)
        n_labels = record.get('n_labels', [])
        n_properties = record.get('n_properties', {})
        m_id = record.get('m_id', None)
        m_labels = record.get('m_labels', [])
        m_properties = record.get('m_properties', {})
        r_type = record.get('r_type', None)
        r_properties = record.get('r_properties', [])

        if n_id:
            nodes.append({
                "id": n_id,
                "label": n_labels[0] if n_labels else "none",
                "properties": n_properties
            })
        if m_id:
            nodes.append({
                "id": m_id,
                "label": m_labels[0] if m_labels else "none",
                "properties": m_properties
            })
        if r_type:
            edges.append({
                "source": n_id,
                "target": m_id,
                "type": r_type,
                "properties": r_properties
            })

    return generate_visualization({
        "nodes": nodes,
        "edges": edges,
    })


html = create_html()
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
