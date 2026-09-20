class RAGException(Exception):
    """Base exception for all RAG-related errors."""
    pass

class DocumentRetrievalError(RAGException):
    """Raised when the vector store search fails."""
    pass

class LLMGenerationError(RAGException):
    """Raised when the LLM call fails."""
    pass

class EmptyQuestionError(RAGException):
    """Raised when the user submits an empty or invalid question."""
    pass
