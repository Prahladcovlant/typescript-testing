from fastapi import APIRouter, HTTPException

from app.schemas import (
    ArticleInsight,
    ArticleInsightsRequest,
    ArticleInsightsResponse,
    CorrelateResponse,
    FourierResponse,
    LoanResponse,
    MarketHealthRequest,
    MarketHealthResponse,
    NormalizeResponse,
    ProjectBlueprintRequest,
    ProjectBlueprintResponse,
    RegressionResponse,
    SentimentResponse,
    TaskScheduleResponse,
)
from app.services import data as data_service
from app.services import finance as finance_service
from app.services import signals as signal_service
from app.services import tasks as task_service
from app.services import text as text_service


router = APIRouter()


@router.post("/article-insights", response_model=ArticleInsightsResponse)
def article_insights(request: ArticleInsightsRequest) -> ArticleInsightsResponse:
    vocabulary, document_terms = text_service.tfidf(
        request.articles, request.tfidf_top_k, request.ngram
    )
    insights = []
    for idx, article in enumerate(request.articles):
        summary, sentences, ratio = text_service.summarize(
            article, request.summary_sentences
        )
        sentiment_payload = text_service.sentiment(article)
        keywords, keyword_scores = text_service.keywords(
            article, request.keyword_top_k
        )
        insights.append(
            ArticleInsight(
                summary=summary,
                sentences=sentences,
                compression_ratio=ratio,
                sentiment=SentimentResponse(**sentiment_payload),
                keywords=keywords,
                keyword_scores=keyword_scores,
                tfidf_top_terms=document_terms[idx] if idx < len(document_terms) else [],
            )
        )
    return ArticleInsightsResponse(insights=insights, vocabulary=vocabulary)


@router.post("/market-health", response_model=MarketHealthResponse)
def market_health(request: MarketHealthRequest) -> MarketHealthResponse:
    if len(request.price_series) != len(request.volume_series):
        raise HTTPException(
            status_code=422,
            detail="price_series and volume_series must have equal length",
        )
    if len(request.macro_features) != len(request.macro_targets):
        raise HTTPException(
            status_code=422,
            detail="macro_features rows must match macro_targets length",
        )

    correlation_payload = data_service.correlations(
        request.price_series, request.volume_series
    )
    correlation = CorrelateResponse(**correlation_payload)

    weights, intercept, r_squared, predictions = data_service.linear_regression(
        request.macro_features, request.macro_targets
    )
    regression = RegressionResponse(
        coefficients=weights,
        intercept=intercept,
        r_squared=round(r_squared, 4),
        predictions=predictions,
    )

    loan_base, total_interest, total_paid, payoff_months, amortization = (
        finance_service.loan_schedule(
            request.loan_principal,
            request.loan_rate,
            request.loan_years,
            request.loan_extra_payment,
        )
    )
    loan = LoanResponse(
        monthly_payment=loan_base,
        interest_paid=total_interest,
        total_paid=total_paid,
        payoff_months=payoff_months,
        amortization=amortization,
    )

    price_zscores, _, _ = data_service.normalize(request.price_series)
    volume_zscores, _, _ = data_service.normalize(request.volume_series)
    volatility = sum(abs(p) + abs(v) for p, v in zip(price_zscores, volume_zscores))
    stress_score = min(
        10.0,
        max(0.0, volatility / len(price_zscores) + (1 - regression.r_squared) * 5),
    )

    return MarketHealthResponse(
        correlation=correlation,
        regression=regression,
        loan=loan,
        stress_score=round(stress_score, 4),
    )


@router.post("/project-blueprint", response_model=ProjectBlueprintResponse)
def project_blueprint(
    request: ProjectBlueprintRequest,
) -> ProjectBlueprintResponse:
    if len(request.tasks) != len(request.effort_hours):
        raise HTTPException(
            status_code=422,
            detail="tasks and effort_hours must have the same length",
        )

    order, critical_path = task_service.schedule_tasks(
        request.tasks, request.dependencies
    )
    schedule = TaskScheduleResponse(schedule=order, critical_path=critical_path)

    zscores, scaled, statistics = data_service.normalize(request.effort_hours)
    effort_profile = NormalizeResponse(
        zscores=zscores, min_max_scaled=scaled, statistics=statistics
    )
    peak_idx = max(range(len(zscores)), key=lambda i: zscores[i])
    workload_focus = request.tasks[peak_idx]

    signal_analysis = None
    if request.progress_signal is not None:
        if request.signal_sample_rate is None:
            raise HTTPException(
                status_code=422,
                detail="signal_sample_rate required when progress_signal provided",
            )
        signal_payload = signal_service.fourier_transform(
            request.progress_signal, request.signal_sample_rate
        )
        signal_analysis = FourierResponse(**signal_payload)

    return ProjectBlueprintResponse(
        schedule=schedule,
        effort_profile=effort_profile,
        workload_focus=workload_focus,
        signal_analysis=signal_analysis,
    )

