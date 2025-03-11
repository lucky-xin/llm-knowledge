from langgraph.store.base import BaseStore
from langgraph.store.postgres import PostgresStore

from utils import create_pg_connect_pool


def create_store() -> BaseStore:
    store = PostgresStore(create_pg_connect_pool("langgraph_store"))
    store.setup()
    return store
