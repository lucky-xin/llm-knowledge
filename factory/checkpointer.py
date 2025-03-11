from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres import PostgresSaver

from utils import create_pg_connect_pool


def create_checkpointer() -> BaseCheckpointSaver[str]:
    saver = PostgresSaver(create_pg_connect_pool("langgraph_checkpointer"))
    saver.setup()
    return saver
