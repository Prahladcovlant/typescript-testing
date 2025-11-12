from fastapi import APIRouter

from app.schemas import FourierRequest, FourierResponse
from app.services import signals as signals_service


router = APIRouter()


@router.post("/fourier", response_model=FourierResponse)
def fourier(request: FourierRequest) -> FourierResponse:
    payload = signals_service.fourier_transform(request.signal, request.sample_rate)
    return FourierResponse(**payload)

