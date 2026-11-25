"""Service layer: framework-independent logic used by the FastAPI app, the CLI and tests."""
import io
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import COLUMN_DESCRIPTIONS, SOURCE_COLUMNS
from .evaluate import evaluate_synthetic, public_summary
from .generator import get_generator
from .rules import validate_dataframe
from .source_data import generate_source_dataset

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT / "data" / "source_subscription_data.csv"
MAX_ROWS = 100_000
RESPONSIBLE_USE = (
    "Synthetic data for testing and evaluating churn-prediction models. Records do not describe "
    "real customers. Do not use as evidence about real customer behaviour or for decisions about "
    "individuals; validate any model on real data before deployment."
)


class ChurnSynthService:
    def __init__(self, source_path=DEFAULT_SOURCE, backend="auto", seed=0):
        self.source_path, self.backend, self.seed = Path(source_path), backend, seed
        self.generator, self.source = None, None

    # -- setup -----------------------------------------------------------------------------
    def load(self):
        if not self.source_path.exists():
            self.source_path.parent.mkdir(parents=True, exist_ok=True)
            generate_source_dataset().to_csv(self.source_path, index=False)
        self.source = pd.read_csv(self.source_path, parse_dates=["signup_date", "churn_date"])
        t0 = time.time()
        self.generator = get_generator(self.backend, seed=self.seed).fit(self.source)
        self.fit_seconds = round(time.time() - t0, 3)
        return self

    def _ready(self):
        if self.generator is None:
            self.load()

    # -- public operations -----------------------------------------------------------------
    def generate(self, n_rows=1000, seed=None, target_churn_rate=None):
        self._ready()
        if not 1 <= n_rows <= MAX_ROWS:
            raise ValueError(f"n_rows must be between 1 and {MAX_ROWS}")
        if target_churn_rate is not None and not 0.01 <= target_churn_rate <= 0.99:
            raise ValueError("target_churn_rate must be between 0.01 and 0.99")
        t0 = time.time()
        if target_churn_rate is None:
            df = self.generator.sample(n_rows, seed=seed)
        else:
            df = self.generator.sample_with_churn_rate(n_rows, target_churn_rate, seed=seed)
        validation = validate_dataframe(df)
        meta = {
            "request_id": str(uuid.uuid4()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": self.generator.name,
            "n_rows": len(df),
            "seed": seed,
            "requested_churn_rate": target_churn_rate,
            "achieved_churn_rate": round(float(df["churned"].mean()), 4),
            "source_churn_rate": round(float(self.source["churned"].mean()), 4),
            "source_rows": len(self.source),
            "columns": list(df.columns),
            "generation_seconds": round(time.time() - t0, 3),
            "validation": {"all_rules_passed": validation["all_rules_passed"],
                           "overall_row_pass_rate": validation["overall_row_pass_rate"],
                           "failed_rules": [k for k, v in validation["rules"].items() if not v["passed"]]},
            "responsible_use": RESPONSIBLE_USE,
        }
        return df, meta

    def evaluate(self, n_rows=3000, seed=0):
        self._ready()
        syn = self.generator.sample(n_rows, seed=seed)
        return public_summary(evaluate_synthetic(self.source, syn, None))

    def schema(self):
        return [{"name": c, "description": COLUMN_DESCRIPTIONS[c]} for c in SOURCE_COLUMNS]

    @staticmethod
    def to_records(df: pd.DataFrame):
        out = df.copy()
        for c in ("signup_date", "churn_date"):
            out[c] = pd.to_datetime(out[c]).dt.strftime("%Y-%m-%d")
        return out.astype(object).where(out.notna(), None).to_dict(orient="records")

    @staticmethod
    def to_csv(df: pd.DataFrame) -> str:
        buf = io.StringIO()
        df.to_csv(buf, index=False, date_format="%Y-%m-%d")
        return buf.getvalue()
