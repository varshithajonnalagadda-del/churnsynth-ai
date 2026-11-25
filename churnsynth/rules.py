"""Business-rule validation for subscription / churn records."""
import numpy as np
import pandas as pd

from .config import (
    CHURN_STATUSES, SIGNUP_END, SIGNUP_START, SNAPSHOT_DATE, STATUSES,
    expected_price, lifetime_months,
)

RULE_DESCRIPTIONS = {
    "unique_customer_id": "customer_id is unique",
    "signup_in_range": "signup_date lies within the allowed signup window",
    "price_matches_plan": "monthly_price equals the list price for plan_type and billing_cycle",
    "valid_status": "subscription_status is one of active/paused/cancelled/expired",
    "churn_flag_consistent": "churned == 1 exactly when status is cancelled or expired",
    "tenure_within_lifetime": "0 <= tenure_months <= months between signup and snapshot",
    "active_tenure_full_lifetime": "non-churned customers have tenure equal to their full lifetime",
    "churned_tenure_positive": "churned customers have tenure_months >= 1",
    "churn_date_consistent": "churn_date present only for churned customers and between signup and snapshot",
    "non_negative_counts": "logins, tickets, payment failures and last_login_days_ago are >= 0",
    "session_minutes_range": "avg_session_minutes is within 1..90",
    "binary_flags_valid": "auto_renew, discount_used and churned are 0/1",
}


def _checks(df: pd.DataFrame) -> dict:
    d = df.copy()
    d["signup_date"] = pd.to_datetime(d["signup_date"])
    d["churn_date"] = pd.to_datetime(d["churn_date"])
    life = lifetime_months(d["signup_date"])
    churned = d["churned"] == 1
    cd = d["churn_date"]
    return {
        "unique_customer_id": ~d["customer_id"].duplicated(keep=False),
        "signup_in_range": d["signup_date"].between(SIGNUP_START, SIGNUP_END),
        "price_matches_plan": np.isclose(d["monthly_price"], expected_price(d)),
        "valid_status": d["subscription_status"].isin(STATUSES),
        "churn_flag_consistent": churned == d["subscription_status"].isin(CHURN_STATUSES),
        "tenure_within_lifetime": (d["tenure_months"] >= 0) & (d["tenure_months"] <= life),
        "active_tenure_full_lifetime": churned | (d["tenure_months"] == life),
        "churned_tenure_positive": ~churned | (d["tenure_months"] >= 1),
        "churn_date_consistent": pd.Series(
            np.where(churned, cd.notna() & (cd >= d["signup_date"]) & (cd <= SNAPSHOT_DATE), cd.isna()),
            index=d.index),
        "non_negative_counts": (d[["monthly_logins", "support_tickets", "payment_failures",
                                   "last_login_days_ago"]] >= 0).all(axis=1),
        "session_minutes_range": d["avg_session_minutes"].between(1, 90),
        "binary_flags_valid": d[["auto_renew", "discount_used", "churned"]].isin([0, 1]).all(axis=1),
    }


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Return {rule: {pass_rate, passed, description}} plus an 'overall' pass rate."""
    results, all_ok = {}, None
    for name, ok in _checks(df).items():
        ok = ok.astype(bool)
        results[name] = {
            "pass_rate": float(ok.mean()),
            "passed": bool(ok.all()),
            "description": RULE_DESCRIPTIONS[name],
        }
        all_ok = ok if all_ok is None else (all_ok & ok)
    return {"rules": results, "overall_row_pass_rate": float(all_ok.mean()),
            "all_rules_passed": all(r["passed"] for r in results.values())}
