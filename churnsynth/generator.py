"""Synthetic subscription/churn generators.

Two interchangeable backends share one interface (fit / sample / sample_with_churn_rate):

* SDVGenerator            - uses the Synthetic Data Vault (GaussianCopulaSynthesizer). Used
                            automatically when `sdv` is installed.
* GaussianCopulaGenerator - a dependency-light NumPy/SciPy Gaussian copula (same statistical
                            idea SDV's GaussianCopulaSynthesizer uses). Used as the fallback.

Both learn on a "model frame" (see to_model_frame) and share the same business-rule
finalisation step (from_model_frame), so derived fields (price, churn flag, churn date,
tenure bounds) are always consistent.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from .config import (
    CHURN_STATUSES, EPOCH, SIGNUP_END, SIGNUP_START, SOURCE_COLUMNS,
    expected_price, lifetime_months,
)

NUM_MODEL = ["signup_ord", "tenure_ratio", "monthly_logins", "avg_session_minutes",
             "support_tickets", "payment_failures", "last_login_days_ago"]
CAT_MODEL = ["plan_type", "billing_cycle", "country", "acquisition_channel",
             "auto_renew", "discount_used", "subscription_status"]
FLOAT_MODEL = {"tenure_ratio", "avg_session_minutes"}


# ----------------------------------------------------------------------------- model frame
def to_model_frame(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Reduce a record table to the independent signals worth learning.

    Derived columns (price, churn flag, churn date, customer id) are dropped and rebuilt
    from business rules. Tenure is modelled as a *ratio of the customer's possible
    lifetime*; for non-churned customers tenure is fully determined by signup date, so
    their ratio is replaced by a draw from the churned customers' ratios (it is
    overwritten when records are rebuilt).
    """
    d = df.copy()
    d["signup_date"] = pd.to_datetime(d["signup_date"])
    life = lifetime_months(d["signup_date"]).clip(lower=1)
    ratio = (d["tenure_months"] / life).clip(0, 1).to_numpy()
    churned = d["churned"].to_numpy() == 1
    if churned.any():
        ratio = np.where(churned, ratio, rng.choice(ratio[churned], size=len(d)))
    m = pd.DataFrame({
        "signup_ord": (d["signup_date"] - EPOCH).dt.days.astype(int),
        "tenure_ratio": ratio,
    })
    for c in NUM_MODEL[2:] + CAT_MODEL:
        m[c] = d[c].to_numpy()
    return m[NUM_MODEL + CAT_MODEL]


def from_model_frame(m: pd.DataFrame, start_index: int = 1) -> pd.DataFrame:
    """Rebuild full, rule-consistent records from sampled model-frame values."""
    m = m.reset_index(drop=True)
    signup = (EPOCH + pd.to_timedelta(np.rint(m["signup_ord"]).astype(int), unit="D"))
    signup = signup.clip(SIGNUP_START, SIGNUP_END)
    life = lifetime_months(signup).to_numpy()
    status = m["subscription_status"].astype(str)
    churned = status.isin(CHURN_STATUSES).astype(int).to_numpy()
    ratio = np.clip(m["tenure_ratio"].to_numpy(float), 0, 1)
    tenure = np.where(churned == 1, np.maximum(1, np.rint(ratio * life)), life)
    tenure = np.minimum(tenure, life).astype(int)
    churn_date = pd.Series(signup) + pd.to_timedelta(tenure * 30, unit="D")
    churn_date = churn_date.where(churned == 1, pd.NaT)

    out = pd.DataFrame({
        "customer_id": [f"CUST-{i:06d}" for i in range(start_index, start_index + len(m))],
        "signup_date": signup.to_numpy(),
        "plan_type": m["plan_type"].astype(str),
        "billing_cycle": m["billing_cycle"].astype(str),
    })
    out["monthly_price"] = expected_price(out)
    out["country"] = m["country"].astype(str)
    out["acquisition_channel"] = m["acquisition_channel"].astype(str)
    out["auto_renew"] = m["auto_renew"].astype(int)
    out["discount_used"] = m["discount_used"].astype(int)
    out["tenure_months"] = tenure
    out["monthly_logins"] = np.maximum(0, np.rint(m["monthly_logins"])).astype(int)
    out["avg_session_minutes"] = np.clip(m["avg_session_minutes"].to_numpy(float), 1, 90).round(1)
    out["support_tickets"] = np.maximum(0, np.rint(m["support_tickets"])).astype(int)
    out["payment_failures"] = np.maximum(0, np.rint(m["payment_failures"])).astype(int)
    out["last_login_days_ago"] = np.maximum(0, np.rint(m["last_login_days_ago"])).astype(int)
    out["subscription_status"] = status
    out["churned"] = churned
    out["churn_date"] = churn_date.to_numpy()
    return out[SOURCE_COLUMNS]


# ----------------------------------------------------------------------------- shared API
class _BaseGenerator:
    name = "base"

    def fit(self, df: pd.DataFrame):  # pragma: no cover - interface
        raise NotImplementedError

    def _sample_model(self, n: int, rng: np.random.Generator) -> pd.DataFrame:  # pragma: no cover
        raise NotImplementedError

    def sample(self, n: int, seed: int | None = None) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        return from_model_frame(self._sample_model(n, rng))

    def sample_with_churn_rate(self, n: int, target: float, seed: int | None = None) -> pd.DataFrame:
        """Generate exactly round(target*n) churned records via conditional (rejection) sampling.

        Selecting on the churn label keeps the learned P(features | churn) intact, so the
        generated data stays realistic at the requested churn rate.
        """
        k = int(round(target * n))
        p = min(max(self.base_churn_rate, 0.01), 0.99)
        pool_n = int(max(k / p, (n - k) / (1 - p)) * 1.4) + 200
        rng_seed = 0 if seed is None else seed
        for attempt in range(6):
            pool = self.sample(pool_n, seed=rng_seed + attempt)
            ch, ac = pool[pool["churned"] == 1], pool[pool["churned"] == 0]
            if len(ch) >= k and len(ac) >= n - k:
                break
            pool_n *= 2
        else:
            raise ValueError("Could not reach the requested churn rate; try a less extreme value.")
        out = pd.concat([ch.sample(k, random_state=rng_seed), ac.sample(n - k, random_state=rng_seed)])
        out = out.sample(frac=1, random_state=rng_seed).reset_index(drop=True)
        out["customer_id"] = [f"CUST-{i:06d}" for i in range(1, n + 1)]
        return out


# ----------------------------------------------------------------------------- copula backend
class GaussianCopulaGenerator(_BaseGenerator):
    name = "gaussian-copula (numpy/scipy fallback)"

    def __init__(self, independent: bool = False, seed: int = 0):
        self.independent = independent  # True = ignore column relationships (baseline model)
        self.seed = seed

    def fit(self, df: pd.DataFrame):
        rng = np.random.default_rng(self.seed)
        m = to_model_frame(df, rng)
        self.cols = list(m.columns)
        self.n_source = len(m)
        self.marginals, U = {}, []
        y = m["subscription_status"].isin(CHURN_STATUSES).astype(float)
        for c in self.cols:
            if c in NUM_MODEL:
                x = m[c].to_numpy(float)
                U.append(rankdata(x, method="average") / (len(x) + 1))
                self.marginals[c] = ("num", np.sort(x))
            else:
                # A copula treats categories as ordered, so order them by churn rate: the
                # category <-> churn relationship then becomes monotonic and is preserved.
                rate = y.groupby(m[c]).mean().sort_values(kind="stable")
                cats = rate.index.to_numpy()
                probs = m[c].value_counts(normalize=True).reindex(cats).to_numpy()
                cum = np.cumsum(probs)
                low = np.concatenate([[0.0], cum[:-1]])
                code = pd.Series(np.arange(len(cats)), index=cats).reindex(m[c].to_numpy()).to_numpy()
                U.append(low[code] + rng.random(len(m)) * probs[code])
                self.marginals[c] = ("cat", cats, cum)
        Z = norm.ppf(np.clip(np.column_stack(U), 1e-6, 1 - 1e-6))
        corr = np.corrcoef(Z, rowvar=False)
        w, v = np.linalg.eigh(corr)  # make sure the matrix is a valid correlation matrix
        corr = (v * np.clip(w, 1e-6, None)) @ v.T
        dd = np.sqrt(np.diag(corr))
        self.corr = corr / np.outer(dd, dd)
        status_probs = m["subscription_status"].value_counts(normalize=True)
        self.base_churn_rate = float(sum(status_probs.get(s, 0) for s in CHURN_STATUSES))
        return self

    def _sample_model(self, n, rng):
        d = len(self.cols)
        if self.independent:
            z = rng.standard_normal((n, d))
        else:
            z = rng.multivariate_normal(np.zeros(d), self.corr, size=n)
        u = norm.cdf(z)
        out = {}
        for j, c in enumerate(self.cols):
            spec = self.marginals[c]
            if spec[0] == "num":
                method = "linear" if c in FLOAT_MODEL else "nearest"
                out[c] = np.quantile(spec[1], u[:, j], method=method)
            else:
                idx = np.minimum(np.searchsorted(spec[2], u[:, j], side="right"), len(spec[1]) - 1)
                out[c] = spec[1][idx]
        return pd.DataFrame(out)


# ----------------------------------------------------------------------------- SDV backend
class SDVGenerator(_BaseGenerator):
    """Synthetic Data Vault backend (GaussianCopulaSynthesizer on the same model frame).

    NOTE: written against the SDV 1.x single-table API; not executed in the build
    environment because `sdv` could not be installed there.
    """
    name = "sdv-gaussian-copula"

    def __init__(self, seed: int = 0):
        self.seed = seed

        def fit(self, df: pd.DataFrame):
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer

        rng = np.random.default_rng(self.seed)
        m = to_model_frame(df, rng)
        for c in CAT_MODEL:
            m[c] = m[c].astype(str)

       metadata = SingleTableMetadata()
       metadata.detect_from_dataframes(data=m)

        for c in CAT_MODEL:
            metadata.update_column(column_name=c, table_name="subscriptions", sdtype="categorical")

        self.synth = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=True)
        self.synth.fit(m)
        self.metadata = metadata
        self.n_source = len(m)
        self.base_churn_rate = float(m["subscription_status"].isin(CHURN_STATUSES).mean())
        return self
        for c in CAT_MODEL:
            metadata.update_column(column_name=c, table_name="subscriptions", sdtype="categorical")
        self.synth = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=True)
        self.synth.fit(m)
        self.metadata = metadata
        self.n_source = len(m)
        self.base_churn_rate = float(m["subscription_status"].isin(CHURN_STATUSES).mean())
        return self

    def _sample_model(self, n, rng):
        import random
        random.seed(int(rng.integers(0, 2**31 - 1)))
        np.random.seed(int(rng.integers(0, 2**31 - 1)))
        return self.synth.sample(num_rows=n)


def get_generator(backend: str = "auto", **kwargs) -> _BaseGenerator:
    """backend: 'auto' (SDV if installed, else copula), 'sdv' or 'copula'."""
    if backend in ("auto", "sdv"):
        try:
            import sdv  # noqa: F401
            return SDVGenerator(**{k: v for k, v in kwargs.items() if k == "seed"})
        except ImportError:
            if backend == "sdv":
                raise
    return GaussianCopulaGenerator(**kwargs)
