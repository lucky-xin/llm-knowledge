from llama_index.core import SimpleDirectoryReader

from agent.doc_chat_agent_v2 import create_agent, create_index_vector_stores, init_context

init_context()
vs = create_index_vector_stores()
reader = SimpleDirectoryReader("/tmp/agent")
docs = reader.load_data()
vs.add(docs)
agent = create_agent()
while True:
    q = input("请问我任何关于文章的问题")
    if q:
        stream = agent.stream({"input": q})
        for chunk in stream:
            output = chunk.get("output")
            print(output)
