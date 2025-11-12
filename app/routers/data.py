from fastapi import APIRouter

from app.schemas import (
    CorrelateRequest,
    CorrelateResponse,
    NormalizeRequest,
    NormalizeResponse,
    RegressionRequest,
    RegressionResponse,
)
from app.services import data as data_service


router = APIRouter()


@router.post("/normalize", response_model=NormalizeResponse)
def normalize(request: NormalizeRequest) -> NormalizeResponse:
    zscores, scaled, stats = data_service.normalize(request.values)
    return NormalizeResponse(zscores=zscores, min_max_scaled=scaled, statistics=stats)


@router.post("/regression", response_model=RegressionResponse)
def regression(request: RegressionRequest) -> RegressionResponse:
    weights, intercept, r_squared, predictions = data_service.linear_regression(
        request.features, request.targets
    )
    return RegressionResponse(
        coefficients=weights,
        intercept=intercept,
        r_squared=round(r_squared, 4),
        predictions=predictions,
    )


@router.post("/correlate", response_model=CorrelateResponse)
def correlate(request: CorrelateRequest) -> CorrelateResponse:
    result = data_service.correlations(request.series_a, request.series_b)
    return CorrelateResponse(**result)

