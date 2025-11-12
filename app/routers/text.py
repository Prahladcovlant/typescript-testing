from fastapi import APIRouter

from app.schemas import (
    KeywordsRequest,
    KeywordsResponse,
    SentimentRequest,
    SentimentResponse,
    SummarizeRequest,
    SummarizeResponse,
    TfIdfRequest,
    TfIdfResponse,
)
from app.services import text as text_service


router = APIRouter()


@router.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    summary, sentences, ratio = text_service.summarize(
        request.text, request.max_sentences
    )
    return SummarizeResponse(
        summary=summary,
        sentences=sentences,
        compression_ratio=ratio,
    )


@router.post("/sentiment", response_model=SentimentResponse)
def sentiment(request: SentimentRequest) -> SentimentResponse:
    payload = text_service.sentiment(request.text)
    return SentimentResponse(**payload)


@router.post("/keywords", response_model=KeywordsResponse)
def keywords(request: KeywordsRequest) -> KeywordsResponse:
    keywords, scores = text_service.keywords(request.text, request.top_k)
    return KeywordsResponse(keywords=keywords, scores=scores)


@router.post("/tfidf", response_model=TfIdfResponse)
def tfidf(request: TfIdfRequest) -> TfIdfResponse:
    vocabulary, document_top_terms = text_service.tfidf(
        request.documents, request.top_k, request.ngram
    )
    return TfIdfResponse(vocabulary=vocabulary, document_top_terms=document_top_terms)

