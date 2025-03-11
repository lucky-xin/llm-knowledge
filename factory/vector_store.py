import os

from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.neo4jvector import Neo4jVectorStore
from llama_index.vector_stores.postgres import PGVectorStore


def create_pg_vector_store() -> BasePydanticVectorStore:
    # Create PGVectorStore instance
    store = PGVectorStore.from_params(
        database=os.getenv("SQL_DB"),
        host=os.getenv("SQL_HOST"),
        password=os.getenv("SQL_PWD"),
        user=os.getenv("SQL_USER"),
        port=os.getenv("SQL_PORT"),
        schema_name="llama_index_vector",
        table_name="llamaindex",
        embed_dim=1024,  # openai embedding dimension
        use_halfvec=False  # Enable half precision
    )
    return store

def create_index_vector_stores() -> BasePydanticVectorStore:
    return Neo4jVectorStore(
        url="bolt://localhost:7687",
        username="neo4j",
        password="neo4j5025",
        # Enable half precision
        use_halfvec=False,
        hybrid_search=False,
        # database="llama_index_vector",
        node_label="Chunk",
        index_name="vector",  # 向量索引名称
        embedding_dimension=1024,  # 向量维度（需与模型匹配）
        create_engine_kwargs={}
    )