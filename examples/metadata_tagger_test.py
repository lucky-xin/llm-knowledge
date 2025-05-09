from langchain_community.document_transformers.openai_functions import create_metadata_tagger, OpenAIMetadataTagger
from langchain_core.documents import Document

from factory.llm import LLMFactory, LLMType

schema = {
    "properties": {
        "movie_title": {"type": "string"},
        "critic": {"type": "string"},
        "tone": {
            "type": "string",
            "enum": ["positive", "negative"]
        },
        "rating": {
            "type": "integer",
            "description": "The number of stars the critic rated the movie"
        }
    },
    "required": ["movie_title", "critic", "tone"]
}
llm_factory = LLMFactory(
    llm_type=LLMType.LLM_TYPE_OPENAI,
)
chat_llm = llm_factory.create_chat_llm()

document_transformer: OpenAIMetadataTagger = create_metadata_tagger(schema, chat_llm)
original_documents = [
    Document(
        page_content="Review of The Bee Movie\nBy Roger Ebert\n\nThis is the greatest movie ever made. 4 out of 5 stars."),
    Document(
        page_content="Review of The Godfather\nBy Anonymous\n\nThis movie was super boring. 1 out of 5 stars.",
        metadata={"reliable": False}
    ),
]

enhanced_documents = document_transformer.transform_documents(original_documents)
for doc in enhanced_documents:
    print("id:", doc.id)
    print("metadata:", doc.metadata)
    print("content:", doc.page_content)
    print("type:", doc.type)
