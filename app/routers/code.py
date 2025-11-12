from fastapi import APIRouter

from app.schemas import CodeDependencyRequest, CodeDependencyResponse
from app.services import code as code_service


router = APIRouter()


@router.post("/dependencies", response_model=CodeDependencyResponse)
def dependencies(request: CodeDependencyRequest) -> CodeDependencyResponse:
    payload = code_service.dependency_analysis(request.code)
    return CodeDependencyResponse(**payload)

