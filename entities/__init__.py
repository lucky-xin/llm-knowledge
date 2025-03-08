from pydantic import Field


class File:
    id: int = Field(description="File id")
    name: str = Field(description="File name")
    type: str = Field(description="File type")
    doc_id: str = Field(description="Doc id")
    md5: str = Field(description="File md5")
