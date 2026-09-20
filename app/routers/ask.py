from fastapi import APIRouter, HTTPException
from app.models.schemas import QuestionRequest, AnswerResponse
from app.rag_service import rag_service
from app.exceptions import RAGException, EmptyQuestionError, RateLimitedError
from app.logging_config import logger

router = APIRouter()


@router.post("/ask", response_model=AnswerResponse, tags=["RAG"])
def ask_question(request: QuestionRequest):
    try:
        result = rag_service.ask(request.question)
        return AnswerResponse(answer=result["answer"], images=result["images"])
    except EmptyQuestionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RateLimitedError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except RAGException as e:
        logger.error(f"RAG error while answering: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your question.")
    