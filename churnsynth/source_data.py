"""Builds the 'approved sample dataset' that SDV learns from (Week 1).

The data is itself simulated (no real customers are available for this project) but
follows realistic subscription-lifecycle behaviour: churn is driven by support
tickets, payment failures, auto-renew, billing cycle, plan and acquisition channel,
and churned customers show lower engagement.
"""
import numpy as np
import pandas as pd

from .config import (
    CHANNEL_P, CHANNELS, COUNTRIES, COUNTRY_P, PLANS, SIGNUP_END, SIGNUP_START,
    SOURCE_COLUMNS, SNAPSHOT_DATE, lifetime_months, price_for,
)


def generate_source_dataset(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    span = (SIGNUP_END - SIGNUP_START).days
    signup = SIGNUP_START + pd.to_timedelta(rng.integers(0, span + 1, n), unit="D")

    plan = rng.choice(PLANS, n, p=[0.45, 0.35, 0.20])
    annual_p = np.select([plan == "Premium", plan == "Standard"], [0.40, 0.30], 0.20)
    cycle = np.where(rng.random(n) < annual_p, "annual", "monthly")
    country = rng.choice(COUNTRIES, n, p=COUNTRY_P)
    channel = rng.choice(CHANNELS, n, p=CHANNEL_P)
    auto_renew = (rng.random(n) < np.where(cycle == "annual", 0.85, 0.65)).astype(int)
    discount = (rng.random(n) < 0.25).astype(int)
    tickets = rng.poisson(1.2, n)
    pay_fail = rng.poisson(np.where(cycle == "monthly", 0.35, 0.12))

    # latent churn risk -> churn label
    logit = (
        -1.25 + 0.32 * tickets + 0.75 * pay_fail - 0.75 * auto_renew
        - 0.90 * (cycle == "annual") + 0.35 * (plan == "Basic") - 0.35 * (plan == "Premium")
        + 0.25 * (channel == "paid_search") - 0.30 * (channel == "referral") + 0.20 * discount
    )
    churned = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    life = lifetime_months(pd.Series(signup)).to_numpy()
    ratio = rng.uniform(0.10, 1.0, n)
    tenure = np.where(churned == 1, np.maximum(1, np.rint(ratio * life)), life).astype(int)

    plan_logins = np.select([plan == "Basic", plan == "Standard"], [12, 16], 20)
    logins = rng.poisson(plan_logins * np.where(churned == 1, 0.65, 1.0))
    session = np.clip(rng.normal(24 - 5 * churned + 3 * (plan == "Premium"), 9), 1, 90).round(1)
    last_login = np.where(
        churned == 1, rng.exponential(18, n) + 2, rng.exponential(8, n)
    ).astype(int)

    p_expired = np.where(auto_renew == 0, 0.55, 0.15)
    churn_status = np.where(rng.random(n) < p_expired, "expired", "cancelled")
    active_status = np.where(rng.random(n) < 0.07, "paused", "active")
    status = np.where(churned == 1, churn_status, active_status)

    churn_date = pd.Series(signup) + pd.to_timedelta(tenure * 30, unit="D")
    churn_date = churn_date.where(churned == 1, pd.NaT)

    df = pd.DataFrame({
        "customer_id": [f"SRC-{i:06d}" for i in range(1, n + 1)],
        "signup_date": signup,
        "plan_type": plan,
        "billing_cycle": cycle,
        "monthly_price": [price_for(p, c) for p, c in zip(plan, cycle)],
        "country": country,
        "acquisition_channel": channel,
        "auto_renew": auto_renew,
        "discount_used": discount,
        "tenure_months": tenure,
        "monthly_logins": logins,
        "avg_session_minutes": session,
        "support_tickets": tickets,
        "payment_failures": pay_fail,
        "last_login_days_ago": last_login,
        "subscription_status": status,
        "churned": churned,
        "churn_date": churn_date,
    })
    assert (df["churn_date"].dropna() <= SNAPSHOT_DATE).all()
    return df[SOURCE_COLUMNS]
