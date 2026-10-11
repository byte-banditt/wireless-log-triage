from functools import lru_cache
from datetime import timezone
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import AwareDatetime, BaseModel, Field, field_validator

from .config import DATA_DIR
from .service import TriageService
from .prediction import FailureRiskService

app = FastAPI(title="Wireless Log Triage")


class LogLine(BaseModel):
    timestamp: AwareDatetime
    test_id: str
    protocol: Literal["wifi", "bluetooth", "cellular"]
    component: str
    status: Literal["pass", "fail", "timeout", "retry"]
    error_code: str = ""
    message: str
    rssi: float
    latency_ms: float = Field(ge=0)
    throughput_mbps: float = Field(ge=0)
    retry_count: int = Field(ge=0)

    @field_validator("timestamp")
    @classmethod
    def to_utc(cls, value):
        return value.astimezone(timezone.utc)


class IngestRequest(BaseModel):
    batch_id: str
    logs: list[LogLine]


@lru_cache(maxsize=1)
def get_service():
    return TriageService(DATA_DIR)


class FailureRiskInput(BaseModel):
    # Outcome text, status, error_code and labels are intentionally absent.
    model_config = {"extra": "forbid"}
    timestamp: AwareDatetime
    test_id: str
    protocol: Literal["wifi", "bluetooth", "cellular"]
    component: str = Field(min_length=1)
    retry_count: int = Field(ge=0)
    rssi: float = Field(allow_inf_nan=False)
    latency_ms: float = Field(ge=0, allow_inf_nan=False)
    throughput_mbps: float = Field(ge=0, allow_inf_nan=False)


class FailureRiskRequest(BaseModel):
    logs: list[FailureRiskInput] = Field(min_length=1, max_length=1000)


def get_failure_service():
    # Load independently of Chroma / BGE and avoid stale cached model artifacts.
    try:
        return FailureRiskService(DATA_DIR)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail="Run 'wireless-triage predict-train' first") from error


@app.post("/predict/failure-risk")
def failure_risk(payload: FailureRiskRequest, service: FailureRiskService = Depends(get_failure_service)):
    try:
        return {"predictions": service.predict([row.model_dump(mode="json") for row in payload.logs])}
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/ingest")
def ingest(payload: IngestRequest, service: TriageService = Depends(get_service)):
    try:
        return {"batch_id": payload.batch_id, "logs": service.ingest(payload.batch_id, [row.model_dump(mode="json") for row in payload.logs])}
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.get("/clusters")
def clusters(service: TriageService = Depends(get_service)):
    return {"clusters": service.clusters()}


@app.get("/triage/{batch_id}")
def triage(batch_id: str, service: TriageService = Depends(get_service)):
    try:
        return service.triage(batch_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="batch not found") from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/anomalies")
def anomalies(service: TriageService = Depends(get_service)):
    return {"anomalies": service.anomalies()}
