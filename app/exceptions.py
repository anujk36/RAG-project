class RAGException(Exception):
    """Base exception for all RAG-related errors."""
    pass

class DocumentRetrievalError(RAGException):
    """Raised when the vector store search fails."""
    pass

class LLMGenerationError(RAGException):
    """Raised when the LLM call fails."""
    pass

class RateLimitedError(RAGException):
    """Raised when the LLM provider's rate limit or quota is exceeded."""
    pass

class EmptyQuestionError(RAGException):
    """Raised when the user submits an empty or invalid question."""
    pass

class IndexingError(RAGException):
    """Raised when a document fails to be processed and added to the vector store."""
    pass
