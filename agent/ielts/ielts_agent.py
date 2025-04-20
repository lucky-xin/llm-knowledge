import os
import re
from typing import TypedDict, Annotated, List

import requests
from langchain_core.messages import BaseMessage
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langgraph.graph import add_messages
from lxml import html

from dict import Dict
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


def create_associate_prompt() -> BasePromptTemplate:
    # 指令模板
    instructions = """
你是一个雅思学习助手，根据用户输入单词进行联想，便于单词记忆。
单词：{keyword}

输出格式：
1.如果单词能进行拆分联想则输出格式为：atmo (水汽) + sphere (球体，球形) ——> 大气圈

严格按照格式输出原始文本，不要自己输出markdown文本！
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", instructions),
        ]
    )


def extract_text(ele):
    tc = ele.text_content().strip()
    tmp = " ".join(line.strip() for line in tc.split("\n"))
    tmp = tmp.split('。')[0]
    tmp = tmp.replace('.', '. ')
    processed_text = re.sub(r'([a-zA-Z])([\u4e00-\u9fa5;])', r'\1 \2', tmp)
    processed_text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z;])', r'\1 \2', processed_text)
    return re.sub(r"^\d+\s*", "", processed_text.replace(' ; ', '；'), flags=re.MULTILINE)


def extract(keyword: str) -> str:
    url = f"https://dict.youdao.com/result?word={keyword}&lang=en"
    resp = requests.get(url)
    tree = html.fromstring(resp.content)
    phonetic = tree.xpath(
        '/html/body/div/div/div/div/div[1]/div/div/section/div[1]/div/div/div/div/div[2]/div[2]/span[2]')
    phonetic_text = phonetic[0].text_content().strip()

    semantics_li = tree.xpath('/html/body/div/div/div/div/div[1]/div/div/section/div[2]/div[2]/div/div[1]/ul/li')
    semantics = []
    for semantic in semantics_li:
        semantic_text = semantic.text_content().strip()
        semantic_text = semantic_text.replace('.', '. ')
        semantics.append(semantic_text)
    phrases = []
    phrase_li1 = tree.xpath('/html/body/div/div/div/div/div[1]/div/div/section/div[3]/div[2]/div/div/div/ul/li')
    idx = 1
    for p in phrase_li1:
        phrases.append(f"{idx}.{extract_text(p)}")
        idx += 1

    phrase_li2 = tree.xpath('/html/body/div/div/div/div/div[1]/div/div/section/div[5]/div[2]/div/ul/li')
    for p in phrase_li2:
        phrases.append(f"{idx}.{extract_text(p)}")
        idx += 1

    examples_li = tree.xpath('/html/body/div/div/div/div/div[1]/div/div/section/div[4]/div[2]/div/div[1]/ul/li')
    idx = 1
    examples = []
    for example in examples_li:
        examples.append(f"{idx}.{extract_text(example)}。")
        idx += 1
    return f"""
{keyword} {phonetic_text}
{"\n".join(examples)}
{"\n".join(semantics)}
"""


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

def extract_html():
    start = 18
    end = 20
    chain_associate = create_associate_prompt() | llm
    dictionary = Dict()
    parent_dir = '/agent/htmls'
    for i in range(start, end):
        # 获取目录下所有文件和子目录
        fp = f"{parent_dir}/chapter{i}"
        print(f"---------------{i}------------------------")
        blocks = []
        dirs = [entry for entry in os.listdir(fp)]
        # 按字母顺序排序
        sorted_dirs = sorted(dirs)
        for filename in sorted_dirs:
            filepath = os.path.join(fp, filename)
            if os.path.isfile(filepath):  # 确保是文件，而不是子目录
                with open(filepath, 'r', encoding='utf-8') as file:
                    k = filename.replace('.html', '')
                    content = file.read()
                    text = dictionary.extract(content)
                    if not text:
                        continue

                    res = chain_associate.invoke(input={"keyword": k})
                    blocks.append(text + '[联]' + res + '\n')

        dp = f"../processed/chapter{i}_processed_final.txt"
        with open(dp, 'a', encoding='utf-8') as dst_file:
            for block in blocks:
                dst_file.write(block)
extract_html()