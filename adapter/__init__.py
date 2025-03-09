import hashlib
from typing import Sequence

from langchain_core.documents import Document as LangchainDocument
from llama_index.core.schema import BaseNode, Document as LLamaIndexDocument


class LangchainDocumentAdapter:

    def __call__(self, nodes: Sequence[BaseNode]) -> Sequence[LangchainDocument]:
        """
        Convert LLamaIndex documents to LangChain documents
        """
        documents = []
        for node in nodes:
            metadata = node.metadata
            if "embedding" not in metadata:
                metadata["embedding"] = node.embedding
            documents.append(
                LangchainDocument(
                    id=node.node_id,
                    page_content=node.text,
                    metadata=metadata
                )
            )
        return documents


class LLamIndexDocumentAdapter:

    def __call__(self, docs: Sequence[LangchainDocument]) -> Sequence[BaseNode]:
        """
        Convert LangChain documents to LLamaIndex documents
        """
        documents = []
        for doc in docs:
            metadata = doc.metadata if doc.metadata else {}
            doc_id = doc.id
            if not doc_id:
                doc_id = hashlib.new('md5', doc.page_content.encode('utf-8')).hexdigest()
            documents.append(
                LLamaIndexDocument(
                    doc_id=doc_id,
                    text=doc.page_content,
                    metadata=metadata
                )
            )
        return documents
