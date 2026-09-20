from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="The question to ask the RAG system")


class AnswerResponse(BaseModel):
    answer: str
    images: list[str] = Field(default_factory=list, description="Relative paths, served under /images/, of images relevant to the answer")


class HealthResponse(BaseModel):
    status: str
    service: str


class ErrorResponse(BaseModel):
    error: str
    detail: str