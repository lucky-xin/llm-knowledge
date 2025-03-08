import threading
import uuid

from langchain_core.messages import HumanMessage, AIMessageChunk
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from llama_index.core import SimpleDirectoryReader, Document, Settings
from llama_index.core.ingestion import run_transformations
from llama_index.core.langchain_helpers.agents import IndexToolConfig, LlamaToolkit

from agent.doc_chat_agent_v2 import create_index_vector_stores, init_context, create_vector_store_index, State, \
    create_prompt
from factory.ai_factory import create_glm_chat_ai

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
print("Creating graph...")
# 初始化 MemorySaver 共例
index_configs = [
    IndexToolConfig(
        name="docs",
        description="useful for when you need to answer questions about the documents",
        query_engine=query_engine
    )
]
llama_toolkit = LlamaToolkit(index_configs=index_configs)
tools = llama_toolkit.get_tools()
search_agent = create_react_agent(create_glm_chat_ai(), tools, prompt=create_prompt())

workflow = StateGraph(State)


def search_chain(state: State):
    resp = search_agent.invoke(state)
    return resp


workflow.add_node("searcher", search_chain)

workflow.add_edge(START, "searcher")
workflow.add_edge("searcher", END)
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
run_id = str(uuid.uuid4())

config = {
    "recursion_limit": 50,
    "configurable": {
        "run_id": run_id,
        "thread_id": str(threading.current_thread().ident)
    },
    # "callbacks": [st_cb]
}
while True:
    q = input("请问我任何关于文章的问题")
    if q:
        stream = graph.stream(
            input={"messages": [HumanMessage(content=q)]},
            config=config,
            stream_mode="messages"
        )
        collected_messages = ""
        for chunks in stream:
            for chunk in chunks:
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    collected_messages += chunk.content
                    print(collected_messages + "▌")
