import re

from lxml import html


def extract_xpath_values(content: str, xpaths: list):
    """
    根据给定的XPath表达式从HTML中提取内容

    参数:
        content: HTML字符串或文件内容
        xpaths: XPath表达式列表

    返回:
        包含每个XPath对应结果的字典
    """
    # 解析HTML
    tree = html.fromstring(content)

    results = {}

    for i, xpath in enumerate(xpaths, 1):
        try:
            # 使用XPath查找元素
            elements = tree.xpath(xpath)

            # 提取元素文本内容
            values = [elem.text_content().strip() for elem in elements if elem.text_content().strip()]

            results[f'xpath_{i}'] = {
                'xpath': xpath,
                'values': values,
                'count': len(values)
            }
        except Exception as e:
            results[f'xpath_{i}'] = {
                'xpath': xpath,
                'error': str(e)
            }

    return results


# 示例使用
if __name__ == "__main__":
    # 你的XPath列表
    xpath_list = [
        '/html/body/div/div/div/div/div[1]/div/div/section/div[1]/div/div/div/div/div[3]/div[2]/span[2]',
        '/html/body/div/div/div/div/div[1]/div/div/section/div[2]/div[2]/div/div[1]/ul/li',
        '/html/body/div/div/div/div/div[1]/div/div/section/div[3]/div[2]/div/div/div/ul/li',
        '/html/body/div/div/div/div/div[1]/div/div/section/div[4]/div[2]/div/div[1]/ul/li',
        '/html/body/div/div/div/div/div[1]/div/div/section/div[5]/div[2]/div/ul/li'
    ]

    # 这里应该是你的HTML内容，可以是字符串或从文件读取
    # 示例中使用的是假设的HTML内容
    with open('2.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    tree = html.fromstring(html_content)
    phonetic = tree.xpath(
        '/html/body/div/div/div/div/div[1]/div/div/section/div[1]/div/div/div/div/div[2]/div[2]/span[2]')
    phonetic_text = phonetic[0].text_content().strip()
    semantics = tree.xpath('/html/body/div/div/div/div/div[1]/div/div/section/div[2]/div[2]/div/div[1]/ul/li')
    for semantic in semantics:
        semantic_text = semantic.text_content().strip()
        semantic_text = semantic_text.replace('.', '. ')
        print(semantic_text)
    phrases = []
    phrase = tree.xpath('/html/body/div/div/div/div/div[1]/div/div/section/div[3]/div[2]/div/div/div/ul/li')
    for p in phrase:
        phrase_text = p.text_content().strip()
        result = " ".join(line.strip() for line in phrase_text.split("\n"))
        tmp = result.replace(' ; ', '；')
        ph = re.sub(r"^\d+\s*", "", tmp, flags=re.MULTILINE)
        print(ph)
        phrases.append(ph)

    examples = []
    examples_li = tree.xpath('/html/body/div/div/div/div/div[1]/div/div/section/div[4]/div[2]/div/div[1]/ul/li')
    for example in examples_li:
        example_text = example.text_content().strip()
        result = " ".join(line.strip() for line in example_text.split("\n"))
        tmp = result.replace(' ; ', '；')
        ex = re.sub(r"^\d+\s*", "", tmp, flags=re.MULTILINE)
        print(ex)
        examples.append(ex)
    li4 = tree.xpath('/html/body/div/div/div/div/div[1]/div/div/section/div[5]/div[2]/div/ul/li')
    # 提取内容
    extracted_data = extract_xpath_values(html_content, xpath_list)

    # 打印结果
    for key, data in extracted_data.items():
        print(f"\n{key}: {data['xpath']}")
        if 'values' in data:
            print(f"找到 {data['count']} 个结果:")
            for i, value in enumerate(data['values'], 1):
                print(f"  {i}. {value}")
        else:
            print(f"错误: {data['error']}")
