import re

from lxml import html


def extract_text(ele):
    tc = ele.text_content().strip()
    r = " ".join(line.strip() for line in tc.split("\n"))
    return re.sub(r"^\d+\s*", "", r.replace(' ; ', '；'), flags=re.MULTILINE)


def extract(keyword: str, content: str) -> str:
    tree = html.fromstring(content)
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
    idx = 0
    for p in phrase_li1:
        phrases.append(f"{idx}.{extract_text(p)}")
        idx += 1

    phrase_li2 = tree.xpath('/html/body/div/div/div/div/div[1]/div/div/section/div[5]/div[2]/div/ul/li')
    for p in phrase_li2:
        phrases.append(f"{idx}.{extract_text(p)}")
        idx += 1

    examples_li = tree.xpath('/html/body/div/div/div/div/div[1]/div/div/section/div[4]/div[2]/div/div[1]/ul/li')
    idx = 0
    examples = []
    for example in examples_li:
        examples.append(f"{idx}.{extract_text(example)}")
        idx += 1
    return f"""
{keyword} {phonetic_text}
{examples}
{semantics}
"""


# 示例使用
if __name__ == "__main__":
    # 示例中使用的是假设的HTML内容
    with open('2.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    print(extract("parallel", html_content))
