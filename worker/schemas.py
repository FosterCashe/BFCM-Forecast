"""Pydantic request/response models for worker/app.py."""
from datetime import date, datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel


class RunCreate(BaseModel):
    tenant_id: UUID
    config_json: Optional[dict[str, Any]] = None


class RunOut(BaseModel):
    run_id: UUID
    tenant_id: UUID
    status: str
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    config_json: Optional[dict[str, Any]] = None
    diagnostics_json: Optional[dict[str, Any]] = None


class ResultRow(BaseModel):
    scenario: str
    date: date
    metric: str
    q10: Optional[float] = None
    q25: Optional[float] = None
    q50: Optional[float] = None
    q75: Optional[float] = None
    q90: Optional[float] = None


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
