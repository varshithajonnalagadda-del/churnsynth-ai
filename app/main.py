"""ChurnSynth AI - FastAPI service.

Run:  uvicorn app.main:app --reload      then open http://127.0.0.1:8000/docs
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from churnsynth import __version__  # noqa: E402
from churnsynth.service import MAX_ROWS, RESPONSIBLE_USE, ChurnSynthService  # noqa: E402

service = ChurnSynthService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()  # learn from the approved sample once at startup
    yield


app = FastAPI(
    title="ChurnSynth AI",
    version=__version__,
    description="Generates realistic synthetic subscription and churn datasets for testing "
                "churn-prediction models. " + RESPONSIBLE_USE,
    lifespan=lifespan,
)


class GenerateRequest(BaseModel):
    n_rows: int = Field(1000, ge=1, le=MAX_ROWS, description="Number of customer records to generate")
    seed: Optional[int] = Field(None, ge=0, le=2**31 - 1, description="Set for reproducible output")
    target_churn_rate: Optional[float] = Field(
        None, ge=0.01, le=0.99, description="Exact share of churned customers; omit to use the learned rate")
    output_format: Literal["json", "csv"] = "json"

    model_config = {"json_schema_extra": {"examples": [
        {"n_rows": 500, "seed": 7, "target_churn_rate": 0.3, "output_format": "json"}]}}


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "version": __version__, "model_ready": service.generator is not None}


@app.get("/schema", tags=["system"])
def schema():
    """Columns of the generated dataset."""
    return {"columns": service.schema()}


@app.get("/model-info", tags=["system"])
def model_info():
    g = service.generator
    return {"generator": g.name, "source_rows": service.source.shape[0],
            "source_churn_rate": round(float(service.source["churned"].mean()), 4),
            "fit_seconds": service.fit_seconds, "max_rows_per_request": MAX_ROWS}


@app.post("/generate", tags=["generation"])
def generate(req: GenerateRequest):
    """Generate synthetic subscription churn data (JSON records + metadata, or CSV download)."""
    try:
        df, meta = service.generate(req.n_rows, req.seed, req.target_churn_rate)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if req.output_format == "csv":
        return Response(
            content=service.to_csv(df), media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="synthetic_churn_data.csv"',
                     "X-Request-Id": meta["request_id"],
                     "X-Achieved-Churn-Rate": str(meta["achieved_churn_rate"]),
                     "X-Rules-Passed": str(meta["validation"]["all_rules_passed"])})
    return {"metadata": meta, "records": service.to_records(df)}


@app.post("/evaluate", tags=["quality"])
def evaluate(n_rows: int = 3000, seed: int = 0):
    """Quick quality check of freshly generated data against the source sample."""
    if not 100 <= n_rows <= 20000:
        raise HTTPException(status_code=422, detail="n_rows must be between 100 and 20000")
    return service.evaluate(n_rows, seed)
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(
        Path(__file__).parent / "website" / "static" / "index.html"
    )