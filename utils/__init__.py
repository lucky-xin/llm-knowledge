import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from typing import Sequence, List, Optional, Callable, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j.graphs.graph_document import GraphDocument
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import create_react_agent
from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.langchain_helpers.agents import IndexToolConfig, LlamaToolkit
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core.schema import BaseNode, Document
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.embeddings.dashscope import DashScopeTextEmbeddingModels, DashScopeEmbedding
from psycopg import Connection
from psycopg2.extras import DictRow
from psycopg_pool import ConnectionPool
from pyvis.network import Network
from streamlit.runtime.uploaded_file_manager import UploadedFile

from factory.llm import LLMFactory, LLMType
from transform.transform import CleanCharTransform, IdGenTransform


def create_prompt() -> BasePromptTemplate:
    # 指令模板
    instructions = """你是一个设计用于査询文档来回答问题的代理，你可以使用文档检索工具，
    并优先基于检索内容来回答问题，如果从文档中找不到任何信息用于回答问题，可以通过其他工具搜索答案，如果所有的工具都不能找到答案，则只需返回“抱歉，这个问题我还不知道。”作为答案。
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", instructions),
            ("placeholder", "{messages}"),
        ]
    )


def create_combine_prompt() -> BasePromptTemplate:
    # 指令模板
    instructions = """你是一个设计用于基于文档来回答问题的助理。
    用户问题：{question}
    
    你只能综合以下文档回答用户问题：
    文档1：{vector_data}
    文档2：{graph_data}
    
    如果根据以上文档不能回答用户问题，则只需返回“抱歉，这个问题我还不知道。”作为答案。
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", instructions),
            ("placeholder", "{messages}"),
        ]
    )


# Write uploaded file in temp dir
def write_file(fp: str, content: bytes):
    try:
        with open(fp, 'wb') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error occurred while writing the file: {e}")
        return False


def create_tmp_dir(tmp_dir: str):
    os.makedirs(tmp_dir, exist_ok=True)


def node_id_func(i: int, doc: BaseNode) -> str:
    return f"{doc.node_id}-{i}"


def create_pg_connect_pool(db_name: str = None) -> ConnectionPool:
    # 如果缓存中不存在，则创建新的历史记录并缓存
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0
    }
    sql_user = os.getenv("SQL_USER")
    sql_pwd = os.getenv("SQL_PWD")
    sql_host = os.getenv("SQL_HOST")
    sql_port = os.getenv("SQL_PORT")
    sql_db = db_name if db_name else os.getenv("SQL_DB")
    connection_string = f"postgresql://{sql_user}:{sql_pwd}@{sql_host}:{sql_port}/{sql_db}?sslmode=disable"
    pool = ConnectionPool[Connection[DictRow]](
        conninfo=connection_string,
        max_size=10,
        kwargs=connection_kwargs
    )
    return pool


def db_init():
    connect_pool = create_pg_connect_pool()
    connection = connect_pool.getconn()
    db_names = ["llm_knowledge", "llama_index_vector", "langgraph_store", "langgraph_checkpointer"]
    cursor = connection.execute("SELECT datname FROM pg_database;")
    exists = [row[0] for row in cursor]
    creates = [db_name for db_name in db_names if db_name not in exists]
    for db_name in creates:
        connection.execute(f"create database {db_name};")
    connection.execute("create extension if not exists vector;")
    connection.commit()
    connection.close()


def init() -> None:
    os.environ['SQL_HOST'] = 'localhost'
    os.environ['SQL_USER'] = 'temp_test_user'
    os.environ['SQL_PWD'] = 'test_userpassword'
    os.environ['SQL_DB'] = 'llm_knowledge'
    os.environ['SQL_PORT'] = '5432'
    # Set Qwen2.5 as the language model and set generation config
    llm_factory = LLMFactory(
        llm_type=LLMType.LLM_TYPE_QWENAI,
    )
    Settings.llm = llm_factory.create_llama_index_llm()
    # Settings.chunk_size = 1024
    # Settings.chunk_overlap = 200
    # LlamaIndex默认使用的Embedding模型被替换为百炼的Embedding模型
    # https://help.aliyun.com/zh/model-studio/user-guide/embedding
    Settings.embed_model = DashScopeEmbedding(model_name=DashScopeTextEmbeddingModels.TEXT_EMBEDDING_V3)
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

    db_init()


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
    llm_factory = LLMFactory(
        llm_type=LLMType.LLM_TYPE_QWENAI,
    )
    return create_react_agent(llm_factory.create_chat_llm(), tools, prompt=create_prompt())


def load_documents(uf: UploadedFile) -> Sequence[BaseNode]:
    parent_path = "/tmp/agent/" + str(uuid.uuid4())
    create_tmp_dir(parent_path)
    # writing the file from RAM to the current directory on disk
    file_path = os.path.join(parent_path, uf.name)
    print(f'Writing {uf.name} to {file_path}')
    write_file(file_path, uf.read())
    return SimpleDirectoryReader(parent_path).load_data()


def extract_entities_prompt():
    return ChatPromptTemplate.from_messages(
        [
            SystemMessage(content="You are extracting organization and person entities from the text."),
            HumanMessage(
                content=
                """
                Use the given format to extract information from the following 
                input: {question}"""
            ),
        ]
    )

def convert_to_graph_documents(transformer: LLMGraphTransformer,
                               documents: Sequence[Document],
                               fetch: Callable[
                                   [LLMGraphTransformer, Document, Optional[RunnableConfig]], GraphDocument],
                               config: Optional[RunnableConfig] = None) -> List[GraphDocument]:
    """Convert a sequence of documents into graph documents.

    Args:
        documents (Sequence[Document]): The original documents.
        config: Additional keyword arguments.
        fetch:  转换函数
        transformer:  转换器
    Returns:
        Sequence[GraphDocument]: The transformed documents as graphs.
    """
    results = []
    """使用线程池并发请求并合并结果"""
    with ThreadPoolExecutor(max_workers=64) as executor:
        # 提交任务到线程池
        futures = [executor.submit(fetch, transformer, doc, config) for doc in documents]

        # 按任务提交顺序收集结果

        # 主线程等待所有子线程完成
        wait(futures, return_when=ALL_COMPLETED)
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"任务异常: {e}")
    return results


# 可视化生成函数
def generate_visualization(data: Dict[str, Any]):
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#1E1E1E",
        font_color="white",
        directed=True
    )

    # 设置布局参数
    net.barnes_hut()

    # 添加节点和边
    for node in data["nodes"]:
        properties = node["properties"]
        net.add_node(
            n_id=node["id"],
            label=node["label"],
            title=json.dumps(properties, indent=2) if properties else "",
            color="#4CAF50" if node["label"] == "Person" else "#2196F3",
            shape="dot" if node["label"] == "Entity" else "diamond"
        )

    for edge in data["edges"]:
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


init()
