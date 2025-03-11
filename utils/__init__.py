import os
import uuid
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED
from typing import Sequence, List, Optional, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_neo4j.graphs.graph_document import GraphDocument
from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars
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
from streamlit.runtime.uploaded_file_manager import UploadedFile

from entities import Entities
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
    instructions = """你是一个设计用于査询文档来回答问题的代理。
    你只能基于以下内容回答用户问题：
    向量数据库召回文档：{vector_data}
    图数据库召回文档：{graph_data}
    
    如果所有的工具都不能找到答案，则只需返回“抱歉，这个问题我还不知道。”作为答案。
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


def create_pg_connect_pool(search_path: str) -> ConnectionPool:
    # 如果缓存中不存在，则创建新的历史记录并缓存
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
        "search_path": search_path
    }
    sql_user = os.getenv("SQL_USER")
    sql_pwd = os.getenv("SQL_PWD")
    sql_host = os.getenv("SQL_HOST")
    sql_port = os.getenv("SQL_PORT")
    sql_db = os.getenv("SQL_DB")
    connection_string = f"postgresql://{sql_user}:{sql_pwd}@{sql_host}:{sql_port}/{sql_db}?sslmode=disable"
    return ConnectionPool[Connection[DictRow]](
        conninfo=connection_string,
        max_size=10,
        kwargs=connection_kwargs
    )

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


def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()


def extract_entities(q: str, llm: BaseChatModel = None) -> Entities:
    if not llm:
        llm_factory = LLMFactory(
            llm_type=LLMType.LLM_TYPE_QWENAI,
        )
        llm = llm_factory.create_chat_llm()
    entity_chain = extract_entities_prompt() | llm.with_structured_output(Entities)
    entities = entity_chain.invoke({"question": q})
    return entities


def convert_to_graph_documents(documents: Sequence[Document],
                               fetch: Callable[[Document, Optional[RunnableConfig]], GraphDocument],
                               config: Optional[RunnableConfig] = None) -> List[GraphDocument]:
    """Convert a sequence of documents into graph documents.

    Args:
        documents (Sequence[Document]): The original documents.
        config: Additional keyword arguments.
        fetch:  转换函数
    Returns:
        Sequence[GraphDocument]: The transformed documents as graphs.
    """
    results = []
    """使用线程池并发请求并合并结果"""
    with ThreadPoolExecutor(max_workers=64) as executor:
        # 提交任务到线程池
        futures = [executor.submit(fetch, doc, config) for doc in documents]

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


init()
