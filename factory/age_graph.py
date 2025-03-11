import os

from langchain_community.graphs.age_graph import AGEGraph
from langchain_community.graphs.graph_store import GraphStore


def create_age_graph() -> GraphStore:
    return AGEGraph(
        graph_name="llama_index_graph",
        conf={
            "host": os.getenv("SQL_HOST"),
            "port": "5455",
            "database": "postgres",
            "user": os.getenv("SQL_USER"),
            "password": os.getenv("SQL_PWD")
        }
    )
