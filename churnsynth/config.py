"""Shared constants and helpers for ChurnSynth AI."""
import numpy as np
import pandas as pd

SNAPSHOT_DATE = pd.Timestamp("2026-01-01")
SIGNUP_START = pd.Timestamp("2023-01-01")
SIGNUP_END = pd.Timestamp("2025-10-31")
EPOCH = pd.Timestamp("1970-01-01")

BASE_PRICE = {"Basic": 9.99, "Standard": 14.99, "Premium": 24.99}
ANNUAL_DISCOUNT = 0.80  # annual plans are billed at 80% of the monthly price (per month)
PLANS = list(BASE_PRICE)
CYCLES = ["monthly", "annual"]
COUNTRIES = ["US", "IN", "UK", "DE", "CA", "AU", "BR"]
COUNTRY_P = [0.30, 0.22, 0.12, 0.09, 0.10, 0.07, 0.10]
CHANNELS = ["organic", "paid_search", "social", "referral", "email"]
CHANNEL_P = [0.30, 0.25, 0.20, 0.15, 0.10]
STATUSES = ["active", "paused", "cancelled", "expired"]
CHURN_STATUSES = ("cancelled", "expired")

SOURCE_COLUMNS = [
    "customer_id", "signup_date", "plan_type", "billing_cycle", "monthly_price",
    "country", "acquisition_channel", "auto_renew", "discount_used",
    "tenure_months", "monthly_logins", "avg_session_minutes", "support_tickets",
    "payment_failures", "last_login_days_ago", "subscription_status",
    "churned", "churn_date",
]
NUMERIC_COLS = [
    "tenure_months", "monthly_price", "monthly_logins", "avg_session_minutes",
    "support_tickets", "payment_failures", "last_login_days_ago",
]
CATEGORICAL_COLS = [
    "plan_type", "billing_cycle", "country", "acquisition_channel",
    "auto_renew", "discount_used", "subscription_status",
]

COLUMN_DESCRIPTIONS = {
    "customer_id": "Unique synthetic customer identifier (CUST-000001 ...). Never copied from real data.",
    "signup_date": "Date the subscription started.",
    "plan_type": "Subscription tier: Basic, Standard or Premium.",
    "billing_cycle": "monthly or annual billing.",
    "monthly_price": "Effective price per month; determined by plan_type and billing_cycle.",
    "country": "Customer country code.",
    "acquisition_channel": "Marketing channel that brought the customer in.",
    "auto_renew": "1 if auto-renewal is on, else 0.",
    "discount_used": "1 if a promotional discount was used, else 0.",
    "tenure_months": "Months the customer has been (or was) subscribed.",
    "monthly_logins": "Average logins per month (engagement).",
    "avg_session_minutes": "Average session length in minutes (engagement).",
    "support_tickets": "Number of support tickets raised.",
    "payment_failures": "Number of failed payment attempts.",
    "last_login_days_ago": "Days since last login at snapshot date.",
    "subscription_status": "active, paused, cancelled or expired.",
    "churned": "1 if status is cancelled/expired, else 0 (the churn label).",
    "churn_date": "Date of churn; empty for non-churned customers.",
}


def price_for(plan: str, cycle: str) -> float:
    p = BASE_PRICE[plan]
    return round(p * (ANNUAL_DISCOUNT if cycle == "annual" else 1.0), 2)


def expected_price(df: pd.DataFrame) -> pd.Series:
    base = df["plan_type"].map(BASE_PRICE).astype(float)
    return (base * np.where(df["billing_cycle"] == "annual", ANNUAL_DISCOUNT, 1.0)).round(2)


def lifetime_months(signup: pd.Series) -> pd.Series:
    """Maximum tenure (in whole 30-day months) a customer can have at the snapshot date."""
    return ((SNAPSHOT_DATE - pd.to_datetime(signup)).dt.days // 30).astype(int)
