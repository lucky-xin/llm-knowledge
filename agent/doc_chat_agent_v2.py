import os
import threading
import uuid
from typing import Sequence, TypedDict, List, Annotated

import streamlit as st
import torch
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import add_messages, StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.ingestion import run_transformations, IngestionCache
from llama_index.core.langchain_helpers.agents import LlamaToolkit, IndexToolConfig
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.schema import BaseNode
from llama_index.core.storage.kvstore import SimpleKVStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels
from llama_index.vector_stores.postgres import PGVectorStore
from streamlit.runtime.uploaded_file_manager import UploadedFile

from callback.streamlit_callback_utils import get_streamlit_cb
from factory.ai_factory import create_llama_index_llm, create_chat_ai
from transform.transform import IdGenTransform, CleanCharTransform

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# Write uploaded file in temp dir
def write_file(fp: str, content: bytes):
    try:
        with open(fp, 'wb') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
        return False


# clear the chat history from streamlit session state
def clear_history():
    pass
    # if 'history' in st.session_state:
    # del st.session_state['history']


def create_tmp_dir(tmp_dir: str):
    os.makedirs(tmp_dir, exist_ok=True)


# @st.cache_resource(ttl="1d")
def create_prompt() -> BasePromptTemplate:
    # 指令模板
    instructions = """你是一个设计用于査询文档来回答问题的代理您可以使用文档检索工具，
    并优先基于检索内容来回答问题，如果从文档中找不到任何信息用于回答问题，可以通过其他工具搜索答案，如果所有的工具都不能找到答案，则只需返回“抱歉，这个问题我还不知道。”作为答案。
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", instructions),
            ("placeholder", "{messages}"),
        ]
    )


def create_index_vector_stores() -> BasePydanticVectorStore:
    # Create PGVectorStore instance
    return PGVectorStore.from_params(
        host=os.getenv("SQL_HOST"),
        port=os.getenv("SQL_PORT"),
        database="llama_index_vector",
        password=os.getenv("SQL_PWD"),
        user=os.getenv("SQL_USER"),
        # openai embedding dimension
        embed_dim=1536,
        # Enable half precision
        use_halfvec=False,
        create_engine_kwargs={

        }
    )


def node_id_func(i: int, doc: BaseNode) -> str:
    return f"{doc.node_id}-{i}"


def init_context():
    # Set Qwen2.5 as the language model and set generation config
    Settings.llm = create_llama_index_llm()
    Settings.chunk_size = 1024
    Settings.chunk_overlap = 200
    # LlamaIndex默认使用的Embedding模型被替换为百炼的Embedding模型
    Settings.embed_model = DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V2)
    # Set the size of the text chunk for retrieval
    sentence_window_parse = SentenceWindowNodeParser(id_func=node_id_func)
    sentence_splitter_parse = SentenceSplitter(id_func=node_id_func)
    Settings.node_parser = sentence_window_parse
    Settings.text_splitter = sentence_window_parse
    keyword_extractor = KeywordExtractor()

    Settings.transformations = [
        CleanCharTransform(),
        keyword_extractor,
        sentence_splitter_parse,
        IdGenTransform()
    ]


def create_query_engine(vector_store: BasePydanticVectorStore) -> BaseQueryEngine:
    # from_documents 方法包含对文档进行切片与建立索引两个步骤
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        # 指定embedding 模型
        embed_model=Settings.embed_model,
        transformations=Settings.transformations
    )
    hyde = HyDEQueryTransform(include_original=True)
    query_engine = index.as_query_engine(Settings.llm)
    return TransformQueryEngine(query_engine=query_engine, query_transform=hyde)


def create_vector_store_index(vector_store: BasePydanticVectorStore) -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex(
        nodes=[],
        storage_context=storage_context,
        vector_store=vector_store,
        embed_model=Settings.embed_model,
        store_nodes_override=True,
        transformations=Settings.transformations
    )
    return index


def create_agent(base_query_engine: BaseQueryEngine) -> CompiledGraph:
    index_configs = [
        IndexToolConfig(
            name="docs",
            description="useful for when you need to answer questions about the documents",
            query_engine=base_query_engine
        )
    ]
    llama_toolkit = LlamaToolkit(index_configs=index_configs)
    tools = llama_toolkit.get_tools()
    return create_react_agent(create_chat_ai(), tools, prompt=create_prompt())


def search_chain(state: State):
    resp = st.session_state.agent.invoke(state)
    return resp


def init():
    if "settings" not in st.session_state:
        init_context()
        st.session_state.settings = True
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

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


def load_documents(uf: UploadedFile) -> Sequence[BaseNode]:
    parent_path = "/tmp/agent/" + str(uuid.uuid4())
    create_tmp_dir(parent_path)
    # writing the file from RAM to the current directory on disk
    file_path = os.path.join(parent_path, uf.name)
    print(f'Writing {uf.name} to {file_path}')
    write_file(file_path, uf.read())
    return SimpleDirectoryReader(parent_path).load_data()


def add_documents(docs: Sequence[BaseNode]) -> None:
    new_nodes = run_transformations(
        nodes=docs,
        transformations=Settings.transformations
    )
    node_ids = [node.node_id for node in new_nodes]
    # get the nodes that already exist in the vector store
    exist_nodes = st.session_state.vector_store_index.vector_store.get_nodes(node_ids=node_ids)
    # nodes already exist in the vector store
    exist_ids = [node.node_id for node in exist_nodes]
    # nodes that need to be inserted into the vector store
    inserts = [node for node in new_nodes if node.node_id not in exist_ids]
    if len(inserts):
        # insert the nodes that need to be inserted into the vector store
        st.session_state.vector_store_index.insert_nodes(inserts)


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
