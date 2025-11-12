from __future__ import annotations

from typing import List, Tuple


def loan_schedule(
    principal: float, annual_rate: float, years: int, extra_payment: float
) -> Tuple[float, float, float, int, List[dict]]:
    monthly_rate = annual_rate / 12 / 100
    months = years * 12
    base_payment = (
        principal
        * (monthly_rate * (1 + monthly_rate) ** months)
        / ((1 + monthly_rate) ** months - 1)
        if monthly_rate > 0
        else principal / months
    )
    monthly_payment = base_payment + extra_payment

    balance = principal
    amortization = []
    total_interest = total_paid = 0.0
    month = 0

    while balance > 0 and month < months + 120:
        month += 1
        interest = balance * monthly_rate
        principal_payment = monthly_payment - interest
        if principal_payment > balance:
            principal_payment = balance
            monthly_payment = interest + principal_payment
        balance -= principal_payment
        total_interest += interest
        total_paid += monthly_payment
        amortization.append(
            {
                "month": month,
                "payment": round(monthly_payment, 2),
                "principal": round(principal_payment, 2),
                "interest": round(interest, 2),
                "balance": round(max(balance, 0), 2),
            }
        )
        if balance <= 1e-6:
            balance = 0
            break

    return (
        round(base_payment, 2),
        round(total_interest, 2),
        round(total_paid, 2),
        month,
        amortization,
    )

