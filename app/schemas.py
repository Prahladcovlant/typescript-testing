from typing import List, Optional

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Full text to summarize")
    max_sentences: int = Field(
        3, ge=1, le=10, description="Maximum number of sentences in the summary"
    )


class SummarizeResponse(BaseModel):
    summary: str
    sentences: List[str]
    compression_ratio: float


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=3)


class SentimentResponse(BaseModel):
    label: str
    polarity: float
    positive_score: float
    negative_score: float
    neutral_score: float
    top_positive_terms: List[str]
    top_negative_terms: List[str]


class KeywordsRequest(BaseModel):
    text: str = Field(..., min_length=10)
    top_k: int = Field(5, ge=1, le=15)


class KeywordsResponse(BaseModel):
    keywords: List[str]
    scores: List[float]


class TfIdfRequest(BaseModel):
    documents: List[str] = Field(..., min_length=2)
    top_k: int = Field(5, ge=1, le=20)
    ngram: int = Field(1, ge=1, le=3)


class TfIdfResponse(BaseModel):
    vocabulary: List[str]
    document_top_terms: List[List[str]]


class NormalizeRequest(BaseModel):
    values: List[float] = Field(..., min_items=2)


class NormalizeResponse(BaseModel):
    zscores: List[float]
    min_max_scaled: List[float]
    statistics: dict


class RegressionRequest(BaseModel):
    features: List[List[float]] = Field(..., min_items=2)
    targets: List[float] = Field(..., min_items=2)


class RegressionResponse(BaseModel):
    coefficients: List[float]
    intercept: float
    r_squared: float
    predictions: List[float]


class LoanRequest(BaseModel):
    principal: float = Field(..., gt=0)
    annual_rate: float = Field(..., gt=0)
    years: int = Field(..., gt=0, le=40)
    extra_payment: float = Field(0, ge=0)


class LoanResponse(BaseModel):
    monthly_payment: float
    interest_paid: float
    total_paid: float
    payoff_months: int
    amortization: List[dict]


class ImageProcessRequest(BaseModel):
    image_base64: str


class ImageProcessResponse(BaseModel):
    processed_base64: str
    width: int
    height: int
    mode: str


class HaversineRequest(BaseModel):
    lat1: float
    lon1: float
    lat2: float
    lon2: float


class HaversineResponse(BaseModel):
    distance_km: float
    distance_miles: float


class FourierRequest(BaseModel):
    signal: List[float] = Field(..., min_items=4)
    sample_rate: float = Field(..., gt=0)


class FourierResponse(BaseModel):
    magnitudes: List[float]
    dominant_frequency: float
    energy: float


class TaskScheduleRequest(BaseModel):
    tasks: List[str] = Field(..., min_items=2)
    dependencies: List[List[str]] = Field(default_factory=list)


class TaskScheduleResponse(BaseModel):
    schedule: List[str]
    critical_path: List[str]


class CodeDependencyRequest(BaseModel):
    code: str = Field(..., min_length=5)


class CodeDependencyResponse(BaseModel):
    modules: List[str]
    standard_library: List[str]
    third_party: List[str]
    relative: List[str]


class CorrelateRequest(BaseModel):
    series_a: List[float] = Field(..., min_items=3)
    series_b: List[float] = Field(..., min_items=3)


class CorrelateResponse(BaseModel):
    pearson: float
    spearman: float
    kendall: float


class ArticleInsightsRequest(BaseModel):
    articles: List[str] = Field(..., min_items=1)
    summary_sentences: int = Field(3, ge=1, le=8)
    keyword_top_k: int = Field(5, ge=1, le=10)
    tfidf_top_k: int = Field(5, ge=1, le=10)
    ngram: int = Field(1, ge=1, le=2)


class ArticleInsight(BaseModel):
    summary: str
    sentences: List[str]
    compression_ratio: float
    sentiment: SentimentResponse
    keywords: List[str]
    keyword_scores: List[float]
    tfidf_top_terms: List[str]


class ArticleInsightsResponse(BaseModel):
    insights: List[ArticleInsight]
    vocabulary: List[str]


class MarketHealthRequest(BaseModel):
    price_series: List[float] = Field(..., min_items=5)
    volume_series: List[float] = Field(..., min_items=5)
    macro_features: List[List[float]] = Field(..., min_items=2)
    macro_targets: List[float] = Field(..., min_items=2)
    loan_principal: float = Field(..., gt=0)
    loan_rate: float = Field(..., gt=0)
    loan_years: int = Field(..., gt=0, le=40)
    loan_extra_payment: float = Field(0, ge=0)


class MarketHealthResponse(BaseModel):
    correlation: CorrelateResponse
    regression: RegressionResponse
    loan: LoanResponse
    stress_score: float


class ProjectBlueprintRequest(BaseModel):
    tasks: List[str] = Field(..., min_items=2)
    dependencies: List[List[str]] = Field(default_factory=list)
    effort_hours: List[float] = Field(..., min_items=2)
    progress_signal: Optional[List[float]] = Field(default=None)
    signal_sample_rate: Optional[float] = Field(default=None)


class ProjectBlueprintResponse(BaseModel):
    schedule: TaskScheduleResponse
    effort_profile: NormalizeResponse
    workload_focus: str
    signal_analysis: Optional[FourierResponse] = None


