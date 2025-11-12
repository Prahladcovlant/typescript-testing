from fastapi import APIRouter

from app.schemas import TaskScheduleRequest, TaskScheduleResponse
from app.services import tasks as task_service


router = APIRouter()


@router.post("/schedule", response_model=TaskScheduleResponse)
def schedule(request: TaskScheduleRequest) -> TaskScheduleResponse:
    order, critical = task_service.schedule_tasks(request.tasks, request.dependencies)
    return TaskScheduleResponse(schedule=order, critical_path=critical)

