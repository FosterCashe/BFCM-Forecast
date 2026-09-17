"""Pydantic request/response models for worker/app.py."""
from datetime import date, datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, computed_field

RHAT_THRESHOLD = 1.01


class RunCreate(BaseModel):
    tenant_id: UUID
    config_json: Optional[dict[str, Any]] = None


class RunOut(BaseModel):
    run_id: UUID
    tenant_id: UUID
    tenant_name: Optional[str] = None
    status: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    config_json: Optional[dict[str, Any]] = None
    diagnostics_json: Optional[dict[str, Any]] = None

    @computed_field
    @property
    def rhat_flagged_params(self) -> Optional[list[str]]:
        """Variables whose rhat exceeds RHAT_THRESHOLD or couldn't be computed
        (null in diagnostics). None when the run has no rhat (ADVI, unfinished)."""
        rhat = (self.diagnostics_json or {}).get("rhat")
        if not rhat:
            return None
        return sorted(k for k, v in rhat.items() if v is None or v > RHAT_THRESHOLD)

    @computed_field
    @property
    def rhat_flag(self) -> Optional[bool]:
        flagged = self.rhat_flagged_params
        return None if flagged is None else len(flagged) > 0


class ResultRow(BaseModel):
    scenario: str
    date: date
    metric: str
    q10: Optional[float] = None
    q25: Optional[float] = None
    q50: Optional[float] = None
    q75: Optional[float] = None
    q90: Optional[float] = None


class TotalOut(BaseModel):
    scenario: str
    metric: str
    start_date: date
    end_date: date
    percentiles: list[float]


class Assumption(BaseModel):
    question_key: str
    evidence_tier: str
    summary: str


class TenantOut(BaseModel):
    tenant_id: UUID
    name: str
    currency: str
    revenue_definition: Optional[str] = None
    assumptions: list[Assumption]


class UploadResponse(BaseModel):
    upload_id: UUID
    filename: str
    size_bytes: int
    status: str
    note: str


class ShopifyConnectRequest(BaseModel):
    tenant_id: UUID
    shop_domain: str
    access_token: str


class ShopifyConnectResponse(BaseModel):
    tenant_id: UUID
    shop_domain: str
    status: str
    note: str
