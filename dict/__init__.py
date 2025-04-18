import re

import requests
from lxml import html


def extract_text(ele):
    tc = ele.text_content().strip()
    tmp = " ".join(line.strip() for line in tc.split("\n"))
    tmp = tmp.split('。')[0]
    tmp = tmp.replace('.', '. ')
    processed_text = re.sub(r'([a-zA-Z])([\u4e00-\u9fa5;])', r'\1 \2', tmp)
    processed_text = re.sub(r'([\u4e00-\u9fa5])([a-zA-Z;])', r'\1 \2', processed_text)
    return re.sub(r"^\d+\s*", "", processed_text.replace(' ; ', '；'), flags=re.MULTILINE)


class Dict:
    def __init__(self, url: str = "https://dict.youdao.com/result"):
        self._url = url

    def download(self, keyword: str):
        url = f"{self._url}?word={keyword}&lang=en"
        resp = requests.get(url)
        if resp.status_code == 200:
            with open(f"/tmp/htmls/{keyword}.html", 'wb+') as dst_file:
                dst_file.write(resp.content)

    def extract(self, content: str) -> str:
        # url = f"{self._url}?word={keyword}&lang=en"
        # resp = requests.get(url)
        tree = html.fromstring(content)
        keyword_ele = tree.xpath(
            '/html/body/div/div/div/div/div[1]/div/div/section/div[1]/div/div/div/div/div/text()'
        )
        keyword = ''
        for keyword_ele_ in keyword_ele:
            keyword = keyword + keyword_ele_
        phonetic_paths = [
            '/html/body/div/div/div/div/div[1]/div/div/section/div[1]/div/div/div/div/div[2]/div/span[2]',
            '/html/body/div/div/div/div/div[1]/div/div/section/div[1]/div/div/div/div/div[2]/div[2]/span[2]',
            '/html/body/div/div/div/div/div[1]/div/div/section/div[1]/div/div/div/div/div[3]/div[2]/span[2]'
        ]
        phonetic = None
        for phonetic_path in phonetic_paths:
            phonetic_eles = tree.xpath(phonetic_path)
            if phonetic_eles is None or len(phonetic_eles) == 0:
                continue
            phonetic = phonetic_eles[0].text_content().strip()
            if phonetic:
                break
        if not phonetic:
            print(keyword)
            return None

        semantics_li = tree.xpath('/html/body/div/div/div/div/div[1]/div/div/section/div[2]/div[2]/div/div[1]/ul/li')
        semantics = []
        for semantic in semantics_li:
            semantic_text = semantic.text_content().strip()
            semantic_text = semantic_text.replace('.', '. ')
            if semantic_text.startswith('【名】'):
                continue
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
{keyword.strip()} {phonetic}
{"\n".join(semantics)}
[例]
{"\n".join(examples)}
[搭]
{"\n".join(phrases)}
"""
