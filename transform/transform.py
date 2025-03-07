import hashlib
from abc import ABC
from typing import Any, Sequence

from llama_index.core.schema import TransformComponent, BaseNode


def remove_empty_lines(text: str) -> str:
    # 按行分割，过滤掉空行，再重新用换行符拼接
    lines = [line for line in text.splitlines() if line.strip() != ""]
    return "\n".join(lines)


class CleanCharTransform(TransformComponent, ABC):
    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        for node in nodes:
            text = remove_empty_lines(node.get_content())
            node.set_content(text)
        return nodes


class IdGenTransform(TransformComponent, ABC):
    def __init__(self, /, **data: Any) -> None:
        super().__init__(**data)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs: Any) -> Sequence[BaseNode]:
        for node in nodes:
            node_id = hashlib.new('md5', node.get_content().encode('utf-8')).hexdigest()
            node.node_id = node_id
        return nodes
