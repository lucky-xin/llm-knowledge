from llama_index.core import Settings, Document, SimpleDirectoryReader
from llama_index.core.extractors import KeywordExtractor
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.embeddings.dashscope import DashScopeEmbedding, DashScopeTextEmbeddingModels

from factory.ai_factory import create_llama_index_llm
from processor import KnowledgeProcessor
from transform.transform import CleanCharTransform, IdGenTransform
from utils import node_id_func

# Set the size of the text chunk for retrieval
sentence_window_parse = SentenceWindowNodeParser(id_func=node_id_func)
sentence_splitter_parse = SentenceSplitter(id_func=node_id_func)
keyword_extractor = KeywordExtractor()

Settings.transformations = [
    CleanCharTransform(),
    keyword_extractor,
    sentence_splitter_parse,
    IdGenTransform()
]

reader = SimpleDirectoryReader("/tmp/agent")
documents: list[Document] = reader.load_data()
knowledge_processor = KnowledgeProcessor()

for document in documents:
    resp = knowledge_processor.extract_triples(document.get_content())
    print(resp)
