from functools import lru_cache
from datetime import timezone
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import AwareDatetime, BaseModel, Field, field_validator

from .config import DATA_DIR
from .service import TriageService

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
