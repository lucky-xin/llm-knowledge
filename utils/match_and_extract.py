import re


def extract_context(text, pattern, context_size=100):
    """
    提取正则匹配项及其前后各100个字符的上下文
    :param text: 原始文本
    :param pattern: 正则表达式模式
    :param context_size: 前后截取的字符数（默认100）
    :return: 包含所有匹配项上下文的列表
    """
    results = []
    for match in re.finditer(pattern, text):
        start, end = match.start(), match.end()
        # 计算前后截取的起始和结束位置
        pre_start = max(0, start - context_size)
        post_end = min(len(text), end + context_size)
        # 提取上下文（包含匹配项本身）
        context = text[pre_start:post_end]
        results.append(context)
    return results


# 示例用法
text = "这是一段示例文本，假设需要抽取关键词'示例'前后的100个字符。这里是更多内容..." * 10
pattern = r"示例"  # 替换为你的正则表达式

contexts = extract_context(text, pattern, context_size=100)
for idx, ctx in enumerate(contexts):
    print(f"匹配项 {idx + 1} 的上下文（共 {len(ctx)} 字符）:\n{ctx}\n{'-' * 50}")
