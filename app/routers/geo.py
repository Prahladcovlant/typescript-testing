from fastapi import APIRouter

from app.schemas import HaversineRequest, HaversineResponse
from app.services import geo as geo_service


router = APIRouter()


@router.post("/distance", response_model=HaversineResponse)
def haversine_distance(request: HaversineRequest) -> HaversineResponse:
    payload = geo_service.haversine(
        request.lat1, request.lon1, request.lat2, request.lon2
    )
    return HaversineResponse(**payload)

