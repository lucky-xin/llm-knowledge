import os
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
    instructions = """
你是一个雅思学习助手，根据用户输入单词列表回答问题。
单词列表：{vocabularies}

你要对每个单词或者词组进行如下操作：
1.如果是单词的话，你要给出：美式音标读音（如果有多种读音则都要列出来）；不同词性解释（读音不一样时要给说明）；3个英文例句及解释；搭配用法（如果有的话）；联想记忆；
2.如果是词组，你要输出：美式音标读音；3个例句；

格式输出：
atmosphere / ˈætməsfɪr /
n.（包围地球或其他行星的）大气，大气层；（房间或其他场所内的）空气；气氛，环境；情调，感染力
[例] 
1.The approaching examination created a atmosphere on the campus. 即将到来的考试在校园里制造了一种紧张气氛。
2.The atmosphere at the house soured. 屋子里的气氛不对了。
3.The atmosphere was quite convivial. 气氛很欢快。
[搭] 
1.atmosphere pressure 大气压
2.working atmosphere 工作环境，工作氛围
[联] atmo (水汽) + sphere (球体，球形) ——> 大气圈

严格按照格式输出原始文本，不要自己输出markdown文本！
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", instructions),
        ]
    )


def read_blanks(file_path) -> dict[str, str]:
    b1 = os.path.exists(file_path)
    if not b1:
        return {}
    with open(file_path, 'r', encoding='utf-8') as file:
        # 读取文件所有内容
        content = file.read()
        # 使用两个换行符('\n\n')作为分隔符进行分割，处理Windows下的'\r\n'
        blocks = [block for block in content.split('\n\n') if block]  # 排除掉纯空的块
    res = {}
    for block in blocks:
        keyword = block.split(' /')[0]
        res[keyword.strip()] = block
    return res

llm_factory = LLMFactory(llm_type=LLMType.LLM_TYPE_QWENAI)
llm = llm_factory.create_llm()

chain = create_combine_prompt() | llm
start = 1
end = 23


def process(src: str, sur_path: str, dst_path: str):
    # 读取文件并去除空行
    with open(src, 'r', encoding='utf-8') as src_file:
        keywords = [line.strip() for line in src_file if line.strip()]
    batch = []
    orig_blocks = read_blanks(sur_path)
    blocks = []
    for keyword in keywords:
        if keyword in orig_blocks:
            blocks.append(orig_blocks[keyword])
        else:
            batch.append(keyword)
        if len(batch) == 10:
            try:
                resp = chain.invoke(input={"vocabularies": ",".join(batch)})
                blocks.append(resp)
            except Exception as e:
                print(f"Error: {e}")
            batch = []
    if len(batch) != 0:
        try:
            resp = chain.invoke(input={"vocabularies": ",".join(batch)})
            blocks.append(resp)
        except Exception as e:
            print(f"Error: {e}")
    with open(dst_path, 'a', encoding='utf-8') as dst_file:
        for block in blocks:
            dst_file.write(block)
            dst_file.write('\n\n')


for i in range(start, end):
    print(f"Processing chapter {i}")
    fp = f"../dst/chapter{i}.txt"
    sur = f"../sources/chapter{i}_processed_final.txt"
    dst = f"../processed/chapter{i}_processed_final.txt"
    process(fp, sur, dst)
