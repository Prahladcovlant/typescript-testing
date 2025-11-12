from fastapi import APIRouter

from app.schemas import ImageProcessRequest, ImageProcessResponse
from app.services import image as image_service


router = APIRouter()


@router.post("/invert", response_model=ImageProcessResponse)
def invert_image(request: ImageProcessRequest) -> ImageProcessResponse:
    payload = image_service.invert_image(request.image_base64)
    return ImageProcessResponse(**payload)

