from langchain_neo4j import Neo4jGraph


def create_neo4j_graph() -> Neo4jGraph:
    graph = Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="neo4j5025",
        enhanced_schema=True,
    )
    graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]"
    )
    return graph
