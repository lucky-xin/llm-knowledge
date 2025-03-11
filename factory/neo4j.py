from langchain_neo4j import Neo4jGraph


def create_neo4j_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url="bolt://localhost:7687",
        username="neo4j",
        password="neo4j5025",
        enhanced_schema=True,
    )

