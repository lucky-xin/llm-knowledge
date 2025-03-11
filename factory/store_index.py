from llama_index.core import StorageContext, VectorStoreIndex, Settings, PropertyGraphIndex
from llama_index.core.data_structs import IndexDict, IndexLPG
from llama_index.core.indices.base import BaseIndex
from llama_index.core.indices.property_graph import DynamicLLMPathExtractor
from llama_index.core.vector_stores.types import BasePydanticVectorStore


def create_vector_store_index(vector_store: BasePydanticVectorStore) -> VectorStoreIndex:
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    index = VectorStoreIndex(
        nodes=[],
        storage_context=storage_context,
        vector_store=vector_store,
        embed_model=Settings.embed_model,
        store_nodes_override=True,
        transformations=Settings.transformations,
        insert_batch_size=10
    )
    return index


def create_graph_store_index(vector_store: BasePydanticVectorStore) -> BaseIndex[IndexLPG]:
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    index = PropertyGraphIndex(
        nodes=[],
        llm=Settings.llm,
        kg_extractors=[DynamicLLMPathExtractor(llm=Settings.llm)],
        storage_context=storage_context,
        vector_store=vector_store,
        embed_model=Settings.embed_model,
        store_nodes_override=True,
        transformations=Settings.transformations,
        insert_batch_size=10
    )
    return index
