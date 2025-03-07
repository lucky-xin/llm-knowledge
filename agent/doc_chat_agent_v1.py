import textwrap

from llama_index.core import SimpleDirectoryReader, Document, Settings
from llama_index.core.ingestion import run_transformations

from agent.doc_chat_agent_v2 import create_index_vector_stores, init_context, create_vector_store_index

init_context()

reader = SimpleDirectoryReader("/tmp/agent")
documents: list[Document] = reader.load_data()

vector_store = create_index_vector_stores()
index = create_vector_store_index(vector_store)

nodes = run_transformations(
    nodes=documents,
    transformations=Settings.transformations
)

node_ids = [node.node_id for node in nodes]
exist_nodes = index.vector_store.get_nodes(node_ids=node_ids)
exist_ids = [node.node_id for node in exist_nodes]

inserts = [node for node in nodes if node.node_id not in exist_ids]
if len(inserts):
    index.insert_nodes(nodes)
query_engine = index.as_query_engine()
# storage_context = StorageContext.from_defaults(vector_store=vector_store)
# index = VectorStoreIndex.from_documents(
#     documents, storage_context=storage_context, show_progress=True
# )


# agent = create_agent()
while True:
    q = input("请问我任何关于文章的问题")
    if q:
        response = query_engine.query(q)
        print(textwrap.fill(str(response), 100))
