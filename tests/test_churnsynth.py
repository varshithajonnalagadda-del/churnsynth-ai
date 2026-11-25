import pandas as pd
import pytest

from churnsynth.evaluate import evaluate_synthetic
from churnsynth.generator import GaussianCopulaGenerator
from churnsynth.rules import validate_dataframe
from churnsynth.service import ChurnSynthService
from churnsynth.source_data import generate_source_dataset


@pytest.fixture(scope="module")
def source():
    return generate_source_dataset(3000, seed=1)


@pytest.fixture(scope="module")
def gen(source):
    return GaussianCopulaGenerator(seed=0).fit(source)


@pytest.fixture(scope="module")
def service(tmp_path_factory, source):
    path = tmp_path_factory.mktemp("d") / "source.csv"
    source.to_csv(path, index=False)
    return ChurnSynthService(source_path=path, backend="copula").load()


def test_source_data_obeys_business_rules(source):
    assert validate_dataframe(source)["all_rules_passed"]


def test_generated_rows_obey_business_rules(gen):
    res = validate_dataframe(gen.sample(5000, seed=3))
    assert res["all_rules_passed"], [k for k, v in res["rules"].items() if not v["passed"]]


def test_shape_and_columns(gen, source):
    out = gen.sample(250, seed=1)
    assert len(out) == 250 and list(out.columns) == list(source.columns)


def test_seed_reproducible(gen):
    pd.testing.assert_frame_equal(gen.sample(300, seed=9), gen.sample(300, seed=9))


def test_ids_unique_and_synthetic(gen):
    out = gen.sample(500, seed=2)
    assert out["customer_id"].is_unique and out["customer_id"].str.startswith("CUST-").all()


@pytest.mark.parametrize("target", [0.05, 0.25, 0.6])
def test_target_churn_rate_exact(gen, target):
    out = gen.sample_with_churn_rate(1000, target, seed=4)
    assert len(out) == 1000 and abs(out["churned"].mean() - target) < 0.001
    assert validate_dataframe(out)["all_rules_passed"]


def test_copula_preserves_relationships_better_than_independent(source):
    real_train = source.iloc[:2000]
    good = GaussianCopulaGenerator(seed=0).fit(real_train).sample(2000, seed=1)
    base = GaussianCopulaGenerator(independent=True, seed=0).fit(real_train).sample(2000, seed=1)
    g = evaluate_synthetic(real_train, good)["correlation"]["mean_abs_corr_diff"]
    b = evaluate_synthetic(real_train, base)["correlation"]["mean_abs_corr_diff"]
    assert g < b


def test_quality_and_privacy(gen, source):
    res = evaluate_synthetic(source, gen.sample(3000, seed=5))
    assert res["statistical_similarity"] > 0.95
    assert res["churn"]["overall_rate_gap"] < 0.03
    assert res["privacy"]["passed"]


def test_service_generate_metadata(service):
    df, meta = service.generate(400, seed=1, target_churn_rate=0.2)
    assert meta["n_rows"] == 400 and meta["validation"]["all_rules_passed"]
    assert meta["achieved_churn_rate"] == pytest.approx(0.2, abs=0.002)
    assert "responsible_use" in meta


@pytest.mark.parametrize("kwargs", [{"n_rows": 0}, {"n_rows": 10**7}, {"n_rows": 10, "target_churn_rate": 1.5}])
def test_service_rejects_bad_input(service, kwargs):
    with pytest.raises(ValueError):
        service.generate(**kwargs)


def test_records_are_json_friendly(service):
    df, _ = service.generate(50, seed=1)
    recs = service.to_records(df)
    assert isinstance(recs[0]["signup_date"], str)
    assert all(r["churn_date"] is None or isinstance(r["churn_date"], str) for r in recs)


# ---------------------------------------------------------------- API tests (need fastapi + httpx)
@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_api_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_api_generate_json(client):
    r = client.post("/generate", json={"n_rows": 100, "seed": 1, "target_churn_rate": 0.3})
    body = r.json()
    assert r.status_code == 200 and len(body["records"]) == 100
    assert body["metadata"]["validation"]["all_rules_passed"]


def test_api_generate_csv(client):
    r = client.post("/generate", json={"n_rows": 20, "output_format": "csv"})
    assert r.status_code == 200 and r.text.splitlines()[0].startswith("customer_id")


@pytest.mark.parametrize("payload", [{"n_rows": 0}, {"n_rows": 1000001}, {"target_churn_rate": 2},
                                     {"output_format": "xml"}])
def test_api_validation_errors(client, payload):
    assert client.post("/generate", json=payload).status_code == 422
