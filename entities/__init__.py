from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from pydantic import Field, BaseModel


class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    answer: str
    vector_data: str
    graph_data: str


class File:
    id: int = Field(description="File id")
    name: str = Field(description="File name")
    type: str = Field(description="File type")
    doc_id: str = Field(description="Doc id")
    md5: str = Field(description="File md5")


# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
                    "appear in the text",
    )
