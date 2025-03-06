import os
import uuid

import streamlit as st
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import BasePromptTemplate
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.langchain_helpers.agents import create_llama_agent, LlamaToolkit, IndexToolConfig
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.dashscope import DashScopeEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.postgres import PGVectorStore
from streamlit.runtime.uploaded_file_manager import UploadedFile

from factory.ai_factory import create_llm


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


@st.cache_resource(ttl="1d")
def create_prompt() -> BasePromptTemplate:
    # 指令模板
    instructions = """你是一个设计用于査询文档来回答问题的代理您可以使用文档检索工具，
    并优先基于检索内容来回答问题，如果从文档中找不到任何信息用于回答问题，可以通过其他工具搜索答案，如果所有的工具都不能找到答案，则只需返回“抱歉，这个问题我还不知道。”作为答案。
    你需要以JSON结构返回。JSON结构体包含output字段，output是你给用户返回的内容。
    """
    base_prompt = hub.pull("hwchase17/react")
    return base_prompt.partial(instructions=instructions)


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
        use_halfvec=True
    )


def init_context():
    # Set Qwen2.5 as the language model and set generation config
    Settings.llm = OpenAILike(
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen-turbo-latest",
        is_chat_model=True,
        context_window=30000
    )

    # LlamaIndex默认使用的Embedding模型被替换为百炼的Embedding模型
    Settings.embed_model = DashScopeEmbedding(model_name="text-embedding-v2")
    # Set the size of the text chunk for retrieval
    Settings.transformations = [SentenceSplitter(chunk_size=1024)]


def create_query_engine(vector_store: BasePydanticVectorStore) -> BaseQueryEngine:
    # from_documents 方法包含对文档进行切片与建立索引两个步骤
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        # 指定embedding 模型
        embed_model=Settings.embed_model
    )

    hyde = HyDEQueryTransform(include_original=True)
    query_engine = index.as_query_engine(Settings.llm)
    return TransformQueryEngine(query_engine=query_engine, query_transform=hyde)


def create_agent() -> AgentExecutor:
    msgs = StreamlitChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="output",
        chat_memory=msgs
    )
    index_configs = [
        IndexToolConfig(
            name="docs",
            description="useful for when you need to answer questions about the documents",
            query_engine=create_query_engine(st.session_state.vs)
        )
    ]
    llama_toolkit = LlamaToolkit(index_configs=index_configs)
    return create_llama_agent(
        toolkit=llama_toolkit,
        llm=create_llm(),
        memory=memory,
        verbose=True,
        handle_parsing_errors="没有从知识库检索到相似的内容"
    )


def init():
    if "vs" not in st.session_state:
        st.session_state.vs = create_index_vector_stores()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])


def to_documents(uf: UploadedFile) -> list[Document]:
    parent_path = "/tmp/agent/" + str(uuid.uuid4())
    create_tmp_dir(parent_path)
    # writing the file from RAM to the current directory on disk
    file_path = os.path.join(parent_path, uf.name)
    print(f'Writing {uf.name} to {file_path}')
    write_file(file_path, uf.read())
    return SimpleDirectoryReader(parent_path).load_data()


if __name__ == "__main__":
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
                docs = to_documents(uploaded_file)
                st.session_state.vs.add(docs)

    q = st.chat_input(placeholder="请问我任何关于文章的问题")
    if q:
        st.chat_message("user").markdown(q)
        st.session_state.messages.append({"role": "user", "content": q})
        collected_messages = ""
        with st.chat_message("assistant"):
            output_placeholder = st.empty()
            st_cb = StreamlitCallbackHandler(st.container())
            agent = create_agent()
            stream = agent.stream({"input": q}, config={"callbacks": [st_cb]})
            for chunk in stream:
                if "output" in chunk:
                    collected_messages += chunk.get("output")
                    output_placeholder.markdown(collected_messages + "▌")
            output_placeholder.markdown(collected_messages)
            st.session_state.messages.append({"role": "assistant", "content": collected_messages})
