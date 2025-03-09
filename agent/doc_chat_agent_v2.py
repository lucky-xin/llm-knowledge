import os
import threading
import uuid
from typing import Sequence, Optional

import streamlit as st
import torch
from langchain_core.messages import HumanMessage, AIMessageChunk
from langchain_core.runnables import RunnableConfig
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from llama_index.core import Settings
from llama_index.core.ingestion import run_transformations, IngestionCache
from llama_index.core.schema import BaseNode, Document
from llama_index.core.storage.kvstore import SimpleKVStore

from adapter import LangchainDocumentAdapter
from callback.streamlit_callback_utils import get_streamlit_cb

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

from entities import State
from factory.ai_factory import create_tongyi_chat_ai

from utils import create_index_vector_stores, create_vector_store_index, create_query_engine, create_agent, \
    load_documents, create_neo4j_graph, convert_to_graph_documents


# clear the chat history from streamlit session state
def clear_history():
    pass
    # if 'history' in st.session_state:
    # del st.session_state['history']


def search_chain(state: State):
    resp = st.session_state.agent.invoke(state)
    return resp


def fetch(doc: Document, c: Optional[RunnableConfig] = None):
    return st.session_state.llm_transformer.process_response(doc, c)


def init():
    if "index" not in st.session_state:
        vector_store = create_index_vector_stores()
        st.session_state.vector_store_index = create_vector_store_index(vector_store)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingest_cache" not in st.session_state:
        st.session_state.ingest_cache = IngestionCache(
            cache=SimpleKVStore(),
        )
    if "agent" not in st.session_state:
        print("Creating agent...")
        query_engine = create_query_engine(st.session_state.vector_store_index.vector_store)
        st.session_state.agent = create_agent(query_engine)
    if "neo4j_graph" not in st.session_state:
        neo4j_graph = create_neo4j_graph()
        neo4j_graph.query(
            "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
        )
        st.session_state.neo4j_graph = neo4j_graph
    if "graph" not in st.session_state:
        print("Creating graph...")
        # 初始化 MemorySaver 共例
        workflow = StateGraph(State)
        workflow.add_node("searcher", search_chain)
        workflow.add_edge(START, "searcher")
        workflow.add_edge("searcher", END)
        checkpointer = MemorySaver()
        graph = workflow.compile(checkpointer=checkpointer)
        st.session_state.graph = graph
    if "llm_transformer" not in st.session_state:
        st.session_state.llm_transformer = LLMGraphTransformer(llm=create_tongyi_chat_ai())

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
