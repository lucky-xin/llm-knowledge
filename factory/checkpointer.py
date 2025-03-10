import os

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import DictRow
from psycopg_pool import ConnectionPool


def create_checkpointer() -> BaseCheckpointSaver[str]:
    # memory = MemorySaver()
    # 如果缓存中不存在，则创建新的历史记录并缓存
    connection_kwargs = {
        "autocommit": True,
        "prepare_threshold": 0,
    }
    sql_user = os.getenv("SQL_USER")
    sql_pwd = os.getenv("SQL_PWD")
    sql_url = os.getenv("SQL_URL")
    sql_db = os.getenv("SQL_DB")
    connection_string = f"postgresql://{sql_user}:{sql_pwd}@{sql_url}/{sql_db}?sslmode=disable"
    print("connection_string:", connection_string)
    saver = PostgresSaver(ConnectionPool[Connection[DictRow]](
        conninfo=connection_string,
        max_size=20,
        kwargs=connection_kwargs
    ))
    saver.setup()
    return saver
