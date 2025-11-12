from fastapi import APIRouter

from app.schemas import LoanRequest, LoanResponse
from app.services import finance as finance_service


router = APIRouter()


@router.post("/loan-amortization", response_model=LoanResponse)
def loan_amortization(request: LoanRequest) -> LoanResponse:
    (
        monthly_payment,
        interest_paid,
        total_paid,
        payoff_months,
        amortization,
    ) = finance_service.loan_schedule(
        request.principal,
        request.annual_rate,
        request.years,
        request.extra_payment,
    )
    return LoanResponse(
        monthly_payment=monthly_payment,
        interest_paid=interest_paid,
        total_paid=total_paid,
        payoff_months=payoff_months,
        amortization=amortization,
    )

