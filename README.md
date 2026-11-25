# ChurnSynth AI: Synthetic Subscription Churn Generator

Generates realistic synthetic subscription and churn datasets for testing churn-prediction models.
It learns statistical patterns from an approved sample dataset with the **Synthetic Data Vault (SDV)**
framework, then generates synthetic customer records that preserve the relationships between customer
lifecycle, subscription activity, engagement, payment behaviour and churn outcomes. A **FastAPI** service
exposes generation, validation and metadata.

## Problem statement
Subscription businesses want to test churn-prediction models across varied customer lifecycle scenarios,
but real churn data for new product lines is limited. Synthetic data reflecting realistic lifecycle patterns
lets teams test prediction models before wider deployment.

## Project layout
```
churnsynth/
  config.py        constants, price table, column descriptions
  source_data.py   builds the approved sample dataset (Week 1)
  rules.py         12 business rules + validator
  generator.py     SDV backend + NumPy/SciPy Gaussian-copula backend (Week 2)
  evaluate.py      similarity, validity, churn-pattern, TSTR, privacy checks (Week 3)
  service.py       framework-independent generation/validation/metadata logic
app/main.py        FastAPI service (Week 4)
scripts/run_pipeline.py   runs everything and writes data + reports
tests/             pytest suite (unit + API tests)
data/              source sample, synthetic output, schema, generation metadata
reports/           Week 1 and Week 3 reports, evaluation JSON, figures
```

## Quick start
```bash
pip install -r requirements.txt
python scripts/run_pipeline.py        # builds data/, reports/, figures
pytest                                 # runs tests
uvicorn app.main:app --reload         # API docs at http://127.0.0.1:8000/docs
```
`generator.get_generator("auto")` uses SDV's `GaussianCopulaSynthesizer` when `sdv` is installed and falls back
to the built-in Gaussian copula otherwise. Both use the same preprocessing and business-rule finalisation.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | service and model status |
| GET | `/schema` | output columns and descriptions |
| GET | `/model-info` | generator, source size, learned churn rate |
| POST | `/generate` | generate data (JSON or CSV) with metadata |
| POST | `/evaluate` | quick quality check against the source sample |

```bash
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
  -d '{"n_rows": 500, "seed": 7, "target_churn_rate": 0.30, "output_format": "json"}'

curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" \
  -d '{"n_rows": 5000, "output_format": "csv"}' -o synthetic.csv
```
Request parameters: `n_rows` (1-100,000), `seed` (reproducibility), `target_churn_rate` (0.01-0.99, exact share
of churned customers), `output_format` (`json` or `csv`). Invalid input returns HTTP 422. Every response carries
metadata: request id, timestamp, generator, seed, requested/achieved churn rate, source size, columns,
validation outcome and a responsible-use notice.

## How it works
1. **Prepare:** the sample has customer, subscription, engagement, payment and churn attributes (`data/schema.json`).
2. **Learn:** independent signals (signup date, tenure as a share of possible lifetime, plan, billing cycle, country,
   channel, auto-renew, discount, logins, session length, tickets, payment failures, last login, status) are modelled
   with a Gaussian copula. Categories are ordered by churn rate so category-churn dependence is preserved.
3. **Generate and enforce rules:** derived fields (price, churn flag, tenure bounds, churn date, ids) are rebuilt from
   business rules so records never contradict themselves.
4. **Target churn rate:** conditional (rejection) sampling on the churn label keeps P(features | churn) intact.
5. **Evaluate:** statistical similarity, correlations, rule validity, churn-by-segment patterns, a churn model trained
   on synthetic data and tested on held-out real data (TSTR), and an exact-copy privacy check.

## Responsible use
Synthetic records do not describe real customers. Use them to test pipelines and models, not as evidence about
real customer behaviour or for decisions about individuals. Always validate a model on real data before deployment.
The generator can only reproduce patterns present in its source sample; biases or gaps in that sample carry over.
Generated data is checked for exact copies of source records, but synthetic data is not a formal privacy guarantee.

## Limitations
- The source sample is simulated; connect your own approved dataset by replacing `data/source_subscription_data.csv`
  with the same columns.
- A Gaussian copula captures monotonic pairwise dependence, not complex interactions.
- The SDV backend and the FastAPI app were written against SDV 1.x / FastAPI APIs; if SDV behaves differently in your
  version, use `get_generator("copula")`.
