from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langgraph.graph import add_messages

from factory.llm import LLMFactory, LLMType


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    answer: str
    vector_data: str
    graph_data: str


def create_combine_prompt() -> BasePromptTemplate:
    # 指令模板
    instructions = """你是一个雅思学习助手，根据用户输入单词列表回答问题。
    单词列表：{vocabularies}
    
    你要对每个单词或者词组进行如下操作：
    1.如果是单词的话，你要给出：美式音标读音（如果有多种读音则都要列出来）；不同词性解释（读音不一样时要给说明）；3个英文例句及解释；搭配用法（如果有的话）；联想记忆；
    2.如果是词组，你要输出：美式音标读音；3个例句；
    
    严格按照以下格式输出：
    atmosphere / ˈætməsfɪr /
    n.（包围地球或其他行星的）大气，大气层；（房间或其他场所内的）空气；气氛，环境；情调，感染力。
    [例] 
    1.The approaching examination created a atmosphere on the campus. 即将到来的考试在校园里制造了一种紧张气氛。
    2.The atmosphere at the house soured. 屋子里的气氛不对了。
    3.The atmosphere was quite convivial. 气氛很欢快。
    [搭] 
    1.atmosphere pressure 大气压
    working atmosphere 工作环境，工作氛围。
    [联] atmo (水汽) + sphere (球体，球形) ——> 大气圈。
    
    严格按照格式输出原始文本，不要自己输出markdown文本！
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", instructions),
        ]
    )


llm_factory = LLMFactory(llm_type=LLMType.LLM_TYPE_QWENAI)
llm = llm_factory.create_llm()

chain = create_combine_prompt() | llm
fp = "../dst/chapter1.txt"
dst_path = "../dst/chapter1_processed.txt"
# 读取文件并去除空行
with open(fp, 'r', encoding='utf-8') as src_file:
    lines = [line.strip() for line in src_file if line.strip()]
batch = []
with open(dst_path, 'w', encoding='utf-8') as dst_file:
    for line in lines:
        batch.append(line)
        if len(batch) == 10:
            resp = chain.invoke(input={"vocabularies": ",".join(batch)})
            dst_file.write(resp)
            batch = []
    if len(batch) != 0:
        resp = chain.invoke(input={"vocabularies": ",".join(batch)})
        dst_file.write(resp)
