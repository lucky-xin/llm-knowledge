import os
import threading
import uuid
from typing import Sequence, Optional

import streamlit as st
import torch
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import GraphCypherQAChain
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from llama_index.core import Settings
from llama_index.core.ingestion import run_transformations
from llama_index.core.schema import BaseNode, Document
from llama_index.core.vector_stores import VectorStoreQuery

from adapter import LangchainDocumentAdapter
from callback.streamlit_callback_utils import get_streamlit_cb

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

from entities import State
from factory.age_graph import create_age_graph
from factory.llm import LLMFactory, LLMType
from factory.store_index import create_vector_store_index
from factory.vector_store import create_pg_vector_store

from utils import load_documents, convert_to_graph_documents, create_combine_prompt


# clear the chat history from streamlit session state
def clear_history():
    pass
    # if 'history' in st.session_state:
    # del st.session_state['history']


def searcher_chain(state: State):
    chain = create_combine_prompt() | st.session_state.llm_tongyi
    resp = chain.invoke(state)
    return resp


def sender_chain(state: State):
    graph_data = graph_retriever_chain(state)
    vector_data = vector_store_retriever_chain(state)
    resp = dict(graph_data, **vector_data)
    return resp


def graph_retriever_chain(state: State):
    # 根据用户问题中包含的实体获取图谱数据库中内容
    messages = state.get("messages")
    question = messages[-1].content
    resp = st.session_state.graph_cypher_qa_chain.invoke({"query": question})
    graph_data = resp.get("result")
    return {"graph_data": graph_data, "question": question}


def vector_store_retriever_chain(state: State):
    # 根据向量获取文档
    messages = state.get("messages")
    question = messages[-1].content
    embedding = Settings.embed_model.get_text_embedding(question)

    resp = st.session_state.index.vector_store.query(
        VectorStoreQuery(
            query_embedding=embedding,
            query_str=question
        )
    )
    vector_data = [el.get_content() for el in resp.nodes]
    return {"vector_data": "\n".join(vector_data)}


def fetch(doc: Document, c: Optional[RunnableConfig] = None):
    return st.session_state.llm_transformer.process_response(doc, c)



def init():
    if "llm_tongyi" not in st.session_state:
        llm_factory = LLMFactory(
            llm_type=LLMType.LLM_TYPE_QWENAI,
        )
        st.session_state.llm_tongyi = llm_factory.create_chat_llm()
    if "index" not in st.session_state:
        vector_store = create_pg_vector_store()
        st.session_state.vector_store_index = create_vector_store_index(vector_store)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "neo4j_graph" not in st.session_state:
        neo4j_graph = create_age_graph()
        neo4j_graph.query(
            "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
        )
        st.session_state.neo4j_graph = neo4j_graph
    if "graph" not in st.session_state:
        print("Creating graph...")
        # 初始化 MemorySaver 共例
        workflow = StateGraph(State)
        workflow.add_edge(START, "sender")
        workflow.add_node("sender", sender_chain)
        workflow.add_node("searcher", searcher_chain)

        workflow.add_edge("sender", "searcher")
        workflow.add_edge("searcher", END)
        checkpointer = MemorySaver()
        graph = workflow.compile(checkpointer=checkpointer)
        st.session_state.graph = graph
    if "llm_transformer" not in st.session_state:
        st.session_state.llm_transformer = LLMGraphTransformer(llm=st.session_state.llm_tongyi)

    if "graph_cypher_qa_chain" not in st.session_state:
        st.session_state.graph_cypher_qa_chain = GraphCypherQAChain.from_llm(
            llm=st.session_state.llm_tongyi,
            graph=st.session_state.neo4j_graph,
            verbose=True,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
        )
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


def add_documents(docs: Sequence[BaseNode]) -> None:
    new_nodes = run_transformations(
        nodes=docs,
        transformations=Settings.transformations
    )
    # 插入向量数据
    st.session_state.vector_store_index.insert_nodes(new_nodes)

    # 插入文档中三元组数据
    langchain_document_adapter = LangchainDocumentAdapter()
    langchain_documents = langchain_document_adapter(docs)
    graph_documents = convert_to_graph_documents(langchain_documents, fetch)
    st.session_state.neo4j_graph.add_graph_documents(graph_documents)


init()
# st.image('img.png')
st.subheader('Qwen🤖')
with st.sidebar:
    # file uploader widget
    uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])
    # chunk size number widget
    chunks = st.number_input('Chunk size:', min_value=2000, max_value=4000, value=2000, on_change=clear_history)

    # add data button widget
    add_data = st.button('Add Data', on_click=clear_history)
    if uploaded_file and add_data:  # if the user browsed a file
        with st.spinner('Reading, chunking and embedding file ...'):
            nodes = load_documents(uploaded_file)
            add_documents(nodes)

q = st.chat_input(placeholder="请问我任何关于文章的问题")
if q:
    st.chat_message("user").markdown(q)
    st.session_state.messages.append({"role": "user", "content": q})
    collected_messages = ""
    with st.chat_message("assistant"):
        output_placeholder = st.empty()
        st_cb = get_streamlit_cb(st.container())
        config = {
            "recursion_limit": 50,
            "configurable": {
                "run_id": str(uuid.uuid4()),
                "thread_id": str(threading.current_thread().ident)
            },
            "callbacks": [st_cb]
        }
        stream = st.session_state.graph.stream(
            input={"messages": [HumanMessage(content=q)]},
            config=config,
            stream_mode="messages"
        )
        for chunks in stream:
            for chunk in chunks:
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    collected_messages += chunk.content
                    output_placeholder.markdown(collected_messages + "▌")
        output_placeholder.markdown(collected_messages)
        st.session_state.messages.append({"role": "assistant", "content": collected_messages})
