# Subscription Churn Data Requirements and Dataset Preparation Report (Week 1)

*Project: ChurnSynth AI - Synthetic Subscription Churn Generator*  |  Generated 2026-11-22

## 1. Problem and objective
Subscription businesses want to test churn-prediction models across varied customer lifecycle scenarios,
but real churn data for new product lines is limited. ChurnSynth AI learns statistical patterns from an
**approved sample dataset** with the Synthetic Data Vault (SDV) framework and generates synthetic customer
subscription records that preserve the relationships between customer lifecycle, subscription activity,
engagement, payment behaviour and churn outcomes.

## 2. Customer lifecycle scenarios covered
| Scenario | How it appears in the data |
|---|---|
| New / young customers | `tenure_months` small, recent `signup_date` |
| Loyal long-tenure customers | active, high `monthly_logins`, auto-renew on, annual billing |
| Payment-trouble customers | `payment_failures` > 0, higher churn risk |
| Disengaged customers | low `monthly_logins`/`avg_session_minutes`, high `last_login_days_ago` |
| Support-heavy customers | many `support_tickets`, higher churn risk |
| Voluntary churn | `subscription_status = cancelled` |
| Passive churn | `subscription_status = expired` (more likely when auto-renew is off) |
| Paused customers | `subscription_status = paused` (not churned) |

## 3. Data requirements
**Customer attributes:** `customer_id`, `country`, `acquisition_channel`, `signup_date`.
**Subscription attributes:** `plan_type`, `billing_cycle`, `monthly_price`, `auto_renew`, `discount_used`, `tenure_months`.
**Engagement:** `monthly_logins`, `avg_session_minutes`, `last_login_days_ago`, `support_tickets`.
**Payment:** `payment_failures`.
**Churn indicators:** `subscription_status`, `churned` (label), `churn_date`.

## 4. Source dataset (approved sample)
- **Rows:** 6,000  |  **Columns:** 18  |  **Snapshot date:** 2026-01-01
- **Overall churn rate:** 25.3%  |  **Missing values:** only `churn_date` (empty for non-churned customers, by design)
- The sample is a simulated stand-in for an approved dataset (no real customer data is used), built so that churn depends on support tickets, payment failures, auto-renew, billing cycle, plan and channel.
- File: `data/source_subscription_data.csv`

### Validated schema
| column | dtype | unique | missing | description |
|---|---|---|---|---|
| customer_id | object | 6000 | 0 | Unique synthetic customer identifier (CUST-000001 ...). Never copied from real data. |
| signup_date | datetime64[ns] | 1025 | 0 | Date the subscription started. |
| plan_type | object | 3 | 0 | Subscription tier: Basic, Standard or Premium. |
| billing_cycle | object | 2 | 0 | monthly or annual billing. |
| monthly_price | float64 | 6 | 0 | Effective price per month; determined by plan_type and billing_cycle. |
| country | object | 7 | 0 | Customer country code. |
| acquisition_channel | object | 5 | 0 | Marketing channel that brought the customer in. |
| auto_renew | int64 | 2 | 0 | 1 if auto-renewal is on, else 0. |
| discount_used | int64 | 2 | 0 | 1 if a promotional discount was used, else 0. |
| tenure_months | int64 | 36 | 0 | Months the customer has been (or was) subscribed. |
| monthly_logins | int64 | 34 | 0 | Average logins per month (engagement). |
| avg_session_minutes | float64 | 477 | 0 | Average session length in minutes (engagement). |
| support_tickets | int64 | 7 | 0 | Number of support tickets raised. |
| payment_failures | int64 | 5 | 0 | Number of failed payment attempts. |
| last_login_days_ago | int64 | 96 | 0 | Days since last login at snapshot date. |
| subscription_status | object | 4 | 0 | active, paused, cancelled or expired. |
| churned | int64 | 2 | 0 | 1 if status is cancelled/expired, else 0 (the churn label). |
| churn_date | datetime64[ns] | 628 | 4483 | Date of churn; empty for non-churned customers. |

## 5. Business rules (checked on every dataset, source and synthetic)
|  | rule | pass rate |
|---|---|---|
| unique_customer_id | customer_id is unique | 100.0% |
| signup_in_range | signup_date lies within the allowed signup window | 100.0% |
| price_matches_plan | monthly_price equals the list price for plan_type and billing_cycle | 100.0% |
| valid_status | subscription_status is one of active/paused/cancelled/expired | 100.0% |
| churn_flag_consistent | churned == 1 exactly when status is cancelled or expired | 100.0% |
| tenure_within_lifetime | 0 <= tenure_months <= months between signup and snapshot | 100.0% |
| active_tenure_full_lifetime | non-churned customers have tenure equal to their full lifetime | 100.0% |
| churned_tenure_positive | churned customers have tenure_months >= 1 | 100.0% |
| churn_date_consistent | churn_date present only for churned customers and between signup and snapshot | 100.0% |
| non_negative_counts | logins, tickets, payment failures and last_login_days_ago are >= 0 | 100.0% |
| session_minutes_range | avg_session_minutes is within 1..90 | 100.0% |
| binary_flags_valid | auto_renew, discount_used and churned are 0/1 | 100.0% |

All rules pass on the source data: **True**.

## 6. Exploratory findings
**Churn by plan**

| plan_type | count | churn_rate |
|---|---|---|
| Basic | 2751.000 | 0.311 |
| Premium | 1153.000 | 0.171 |
| Standard | 2096.000 | 0.221 |

**Churn by billing cycle**

| billing_cycle | count | churn_rate |
|---|---|---|
| annual | 1623.000 | 0.105 |
| monthly | 4377.000 | 0.308 |

**Mean behaviour of active vs churned customers**

|  | monthly_logins | avg_session_minutes | support_tickets | payment_failures | last_login_days_ago | tenure_months |
|---|---|---|---|---|---|---|
| Active (0) | 15.120 | 24.586 | 1.108 | 0.221 | 7.563 | 18.850 |
| Churned (1) | 9.401 | 19.143 | 1.508 | 0.529 | 19.711 | 10.338 |

Key takeaways: churned customers log in less, have longer gaps since last login, raise more support tickets and
suffer more payment failures; annual and auto-renew customers churn less. Basic plans churn more than Premium.
These are exactly the relationships the synthetic data must preserve.

## 7. Data-quality requirements for generated data
1. 100% of generated rows satisfy all business rules in section 5.
2. Per-column distributions close to the source (similarity >= 0.95 on average).
3. Correlation structure and churn-by-segment patterns preserved.
4. A churn model trained on synthetic data must still rank real customers sensibly (AUC close to a model trained on real data).
5. No exact copies of source records.

## 8. Synthetic-data generation objectives
Generate any number of records (up to 100,000), reproducibly (seed), optionally at a **target churn rate**,
returning metadata with every dataset. Design details are in `README.md`.
