"""
Precheck AI — Oncology RCM Workspace. SINGLE-FILE backend, SQLite-persisted.

    python -m pip install fastapi uvicorn pydantic
    python -m uvicorn server:app --port 8000

Docs at http://localhost:8000/docs. All patient data is fictional.

Architecture (single file for deployment simplicity, layered internally):
  - Models        (pydantic; the API contract — PRD v1.1 Section 6)
  - Reference     (payer/CPT/NCCN/CARC tables; production = CMS/payer scrapes)
  - Rules         (L0 pre-visit + L1 claim checks — deterministic)
  - Heuristics    (L2 scoring / L3 NLP / L4 letters — labeled stand-ins)
  - Repo          (SQLite repository; PG-portable schema. Swapping to Aurora
                   PostgreSQL means reimplementing ONLY this class.)
  - API           (FastAPI routes; no storage logic inside handlers)

Eligibility, Analytics, and Appeals are computed live from the repository —
no static demo payloads. State (owners, statuses, evidence, denials, audit
log) persists across restarts in precheck.db next to this file.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets as pysecrets
import sqlite3
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

_ALLOW_MODEL_PREFIX = ConfigDict(protected_namespaces=())

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precheck.db")

# Optional API-key gate (PRD 9.3). Unset = open demo mode.
API_KEY = os.environ.get("PRECHECK_API_KEY", "")
ESCALATION_HOURS = 24  # PRD v1.1 Section 12 — unowned exceptions escalate

# Session-token signing secret. CHANGE IN PRODUCTION (set PRECHECK_SECRET).
AUTH_SECRET = os.environ.get("PRECHECK_SECRET", "demo-secret-change-me")
# Login requirement: on by default; set PRECHECK_AUTH=0 for an open demo.
AUTH_REQUIRED = os.environ.get("PRECHECK_AUTH", "1") != "0"
# Demo helper exposing the current TOTP code so the MFA flow is testable
# without an authenticator app. MUST be disabled in production (set =0).
DEMO_MFA_HELPER = os.environ.get("PRECHECK_DEMO_MFA", "1") != "0"
SESSION_HOURS = 12

# =========================================================================
# Models
# =========================================================================

class ExceptionCategory(str, Enum):
    AUTH = "AUTH"; ELIGIBILITY = "ELIGIBILITY"; REFERRAL = "REFERRAL"
    CDX = "CDX"; CODING = "CODING"; COB = "COB"; UNITS = "UNITS"
    CPT = "CPT"; DX = "DX"; MODIFIER = "MODIFIER"


class SourceLayer(str, Enum):
    L0 = "L0"; L1 = "L1"; L2 = "L2"; L3 = "L3"


class ExceptionStatus(str, Enum):
    OPEN = "OPEN"; IN_PROGRESS = "IN_PROGRESS"; RESOLVED = "RESOLVED"; WAIVED = "WAIVED"


class VerificationMethod(str, Enum):
    PORTAL = "PORTAL"; PHONE = "PHONE"; EDI = "EDI"; FAX = "FAX"


class EvidenceCreate(BaseModel):
    payer_source: str
    verification_method: VerificationMethod
    reference_number: Optional[str] = None
    verified_by: str
    attachment_url: Optional[str] = None


class Evidence(EvidenceCreate):
    evidence_id: str = Field(default_factory=lambda: f"EVD-{uuid.uuid4().hex[:8].upper()}")
    exception_id: str
    verified_at: datetime = Field(default_factory=datetime.utcnow)


class ExceptionUpdate(BaseModel):
    owner: Optional[str] = None
    due_date: Optional[date] = None
    status: Optional[ExceptionStatus] = None
    resolution_notes: Optional[str] = None


class ExceptionRecord(BaseModel):
    exception_id: str = Field(default_factory=lambda: f"EXC-{uuid.uuid4().hex[:6].upper()}")
    case_id: str
    category: ExceptionCategory
    source_layer: SourceLayer
    description: str
    blocks_service: bool = True
    owner: str = "Unassigned"
    due_date: Optional[date] = None
    status: ExceptionStatus = ExceptionStatus.OPEN
    resolution_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    escalated: bool = False  # computed: unowned + OPEN for > ESCALATION_HOURS
    evidence: List[Evidence] = []


class PrevisitCheckRequest(BaseModel):
    case_id: str
    patient_ref: str
    payer_id: str
    plan_type: str
    appointment_date: date
    cpt_codes_planned: List[str] = []
    referral_on_file: bool = False
    specialty_code: str = "ONCOLOGY"
    provider_npi: str
    patient_name: Optional[str] = None
    age_sex: Optional[str] = None
    cancer: Optional[str] = None
    regimen: Optional[str] = None
    physician: Optional[str] = None
    appt_time: Optional[str] = None


class PrevisitChecks(BaseModel):
    eligibility: str
    cob: str
    network_status: str
    referral_required: bool
    referral_on_file: bool


class PrevisitCheckResponse(BaseModel):
    case_id: str
    patient_ref: str
    patient_name: Optional[str] = None
    age_sex: Optional[str] = None
    cancer: Optional[str] = None
    regimen: Optional[str] = None
    physician: Optional[str] = None
    appt_time: Optional[str] = None
    payer_id: str = ""
    plan_type: str = ""
    appointment_date: date
    readiness_status: str
    checks: PrevisitChecks
    exceptions_created: List[str]
    next_action: str
    processing_ms: int


class RuleBlocker(BaseModel):
    rule: str
    message: str
    exception_id: str


class ShapReason(BaseModel):
    feature: str
    value: Any
    impact: str
    message: str


class CdxStatus(BaseModel):
    pdl1_present: bool = False
    pdl1_score: Optional[int] = None
    egfr_present: bool = False
    brca_present: bool = False
    her2_present: bool = False


class NccnMatch(BaseModel):
    regimen: Optional[str] = None
    category: Optional[str] = None
    cancer_type: Optional[str] = None


class PrecheckRequest(BaseModel):
    claim_id: str
    payer_id: str
    plan_type: str
    primary_dx: str
    secondary_dx: List[str] = []
    cpt_codes: List[str] = []
    jcodes: List[str] = []
    modifiers: List[List[Optional[str]]] = []
    ndc_code: Optional[str] = None
    billed_units: int = 1
    billed_amount: float = 0.0
    place_of_service: str
    dos: date
    auth_number: Optional[str] = None
    clinical_note: Optional[str] = None
    specialty_code: str = "ONCOLOGY"
    provider_npi: str
    patient_name: Optional[str] = None
    mrn: Optional[str] = None
    age_sex: Optional[str] = None
    stage: Optional[str] = None
    line: Optional[str] = None
    plan_name: Optional[str] = None


class PrecheckResponse(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX
    claim_id: str
    patient_name: Optional[str] = None
    mrn: Optional[str] = None
    age_sex: Optional[str] = None
    stage: Optional[str] = None
    line: Optional[str] = None
    plan_name: Optional[str] = None
    payer_id: str = ""
    payer_name: str = ""
    plan_type: str = ""
    primary_dx: str = ""
    cpt_codes: List[str] = []
    jcodes: List[str] = []
    billed_amount: float = 0.0
    billed_units: int = 0
    dos: Optional[date] = None
    denied: bool = False
    denial_carc: Optional[str] = None
    denial_rarc: Optional[str] = None
    denial_reason: Optional[str] = None
    denied_date: Optional[date] = None
    paid_amount: Optional[float] = None
    risk_score: float
    risk_level: str
    predicted_denial_class: str
    denial_probability: Dict[str, float]
    auth_required: bool
    rule_blockers: List[RuleBlocker]
    shap_reasons: List[ShapReason]
    recommended_actions: List[str]
    cdx_status: CdxStatus
    nccn_match: NccnMatch
    processing_ms: int
    model_version: str


class NlpNotesRequest(BaseModel):
    clinical_note: str
    jcodes: List[str] = []
    primary_dx: Optional[str] = None


class NlpNotesResponse(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX
    cdx_status: CdxStatus
    nccn_match: NccnMatch
    model_version: str


class AppealGenerateRequest(BaseModel):
    claim_id: str
    carc: str


class AppealGenerateResponse(BaseModel):
    model_config = _ALLOW_MODEL_PREFIX
    claim_id: str
    carc: str
    letter_text: str
    cited_evidence: List[str]
    model_version: str


class DenyClaimRequest(BaseModel):
    carc: str
    rarc: Optional[str] = None
    paid_amount: Optional[float] = 0.0


# =========================================================================
# Reference data
# =========================================================================

CPT_JCODE_REFERENCE = {
    "J9271": {"description": "Pembrolizumab (Keytruda) per 1mg", "drug": "pembrolizumab", "auth_required": True},
    "J9299": {"description": "Nivolumab (Opdivo) per 1mg", "drug": "nivolumab", "auth_required": True},
    "J9355": {"description": "Trastuzumab (Herceptin) per 10mg", "drug": "trastuzumab", "auth_required": True},
    "J9023": {"description": "Abiraterone (Zytiga) per 10mg", "drug": "abiraterone", "auth_required": True},
    "J9055": {"description": "Bevacizumab (Avastin) per 10mg", "drug": "bevacizumab", "auth_required": True},
    "J9045": {"description": "Carboplatin per 50mg", "drug": "carboplatin", "auth_required": False},
    "J9312": {"description": "Rituximab per 10mg", "drug": "rituximab", "auth_required": True},
    "96413": {"description": "Chemo infusion, initial, first hour", "drug": None, "auth_required": "SOMETIMES"},
    "96415": {"description": "Chemo infusion, each addl hour", "drug": None, "auth_required": False},
    "96417": {"description": "Each addl sequential infusion", "drug": None, "auth_required": False},
    "88360": {"description": "IHC, per antibody", "drug": None, "auth_required": False},
    "81275": {"description": "KRAS gene analysis", "drug": None, "auth_required": False},
}

NCCN_TABLE = {
    "pembrolizumab": {"regimen": "Pembrolizumab monotherapy", "category": "1", "cancer_types": ["NSCLC"]},
    "nivolumab": {"regimen": "Nivolumab monotherapy", "category": "1", "cancer_types": ["NSCLC", "Melanoma"]},
    "trastuzumab": {"regimen": "Trastuzumab-based therapy", "category": "1", "cancer_types": ["Breast"]},
    "abiraterone": {"regimen": "Abiraterone + prednisone", "category": "1", "cancer_types": ["Prostate"]},
    "bevacizumab": {"regimen": "Bevacizumab combination therapy", "category": "2A", "cancer_types": ["Colorectal", "NSCLC"]},
    "carboplatin": {"regimen": "Carboplatin-based chemotherapy", "category": "1", "cancer_types": ["Ovarian"]},
    "rituximab": {"regimen": "Rituximab-based therapy", "category": "1", "cancer_types": ["Lymphoma"]},
}

CANCER_TYPE_MAP = {"C34": "NSCLC / Lung", "C50": "Breast", "C61": "Prostate", "C18": "Colorectal",
                   "C43": "Melanoma", "C56": "Ovarian", "C71": "Brain", "C85": "Lymphoma"}

PAYER_NAMES = {"UHC01": "UnitedHealthcare", "AETNA1": "Aetna", "BCBS01": "BCBS", "HUM01": "Humana",
               "00430": "Northstar Choice", "MERID1": "Meridian Advantage", "MCO1": "Statewide Medicaid MCO",
               "TRI1": "Tricare", "FHIR01": "FHIR-sourced payer"}

PLAN_KIND = {"COMMERCIAL": "Commercial", "MEDICARE_ADVANTAGE": "Medicare Advantage",
             "MEDICARE_FFS": "Medicare FFS", "MEDICAID": "Medicaid", "GOVERNMENT": "Government"}

REFERRAL_RULES = {
    ("00430", "MEDICARE_ADVANTAGE"): True, ("UHC01", "COMMERCIAL"): True,
    ("AETNA1", "COMMERCIAL"): False, ("BCBS01", "COMMERCIAL"): True,
    ("TRI1", "GOVERNMENT"): True, ("HUM01", "MEDICARE_ADVANTAGE"): True, ("MCO1", "MEDICAID"): True,
}

MOCK_ELIGIBILITY_DB = {
    "CASE-2026-0201": {"eligibility": "ACTIVE", "cob": "RESOLVED", "network_status": "IN_NETWORK",
                       "effective": "2025-01-01", "termed": None},
    "CASE-2026-0202": {"eligibility": "ACTIVE", "cob": "RESOLVED", "network_status": "IN_NETWORK",
                       "effective": "2024-07-01", "termed": None},
    "CASE-2026-0203": {"eligibility": "INACTIVE", "cob": "RESOLVED", "network_status": "IN_NETWORK",
                       "effective": "2025-01-01", "termed": "2026-06-30"},
    "CASE-2026-0204": {"eligibility": "ACTIVE", "cob": "UNRESOLVED", "network_status": "IN_NETWORK", "soft": True,
                       "effective": "2026-01-01", "termed": None},
    "CASE-2026-0205": {"eligibility": "ACTIVE", "cob": "RESOLVED", "network_status": "IN_NETWORK", "soft": True,
                       "effective": "2026-01-01", "termed": None},
    "CASE-2026-0206": {"eligibility": "INACTIVE", "cob": "UNRESOLVED", "network_status": "OUT_OF_NETWORK",
                       "effective": "2024-01-01", "termed": "2026-05-31"},
}
DEFAULT_ELIGIBILITY = {"eligibility": "UNKNOWN", "cob": "UNKNOWN", "network_status": "UNKNOWN",
                       "effective": None, "termed": None}

PAYER_DENIAL_RATE = {"00430": 0.44, "UHC01": 0.31, "AETNA1": 0.22, "BCBS01": 0.27, "HUM01": 0.38, "MCO1": 0.33}
DEFAULT_PAYER_DENIAL_RATE = 0.15

CARC_REFERENCE = {
    "197": {"description": "Precertification/authorization absent", "fix": "Obtain prior authorization before resubmission"},
    "15": {"description": "Payment adjusted — benefits not assigned", "fix": "Verify auth was obtained and linked to claim"},
    "16": {"description": "Claim lacks information or has submission error", "fix": "Attach the missing documentation identified in the RARC and resubmit"},
    "50": {"description": "Service not covered under benefit plan", "fix": "Verify plan coverage for this ICD/CPT combination"},
    "96": {"description": "Non-covered charge — not medically necessary", "fix": "Attach NCCN guideline letter, pathology report"},
    "151": {"description": "Payment adjusted — units/level not supported", "fix": "Provide clinical documentation supporting billed units/level"},
    "167": {"description": "Non-covered injury/illness", "fix": "Review primary dx — may need sequence correction"},
    "4": {"description": "Service inconsistent with covered modifier", "fix": "Review modifier applicability; remove incorrect modifier"},
    "18": {"description": "Duplicate claim/service", "fix": "Verify no duplicate submission; check claim_id"},
    "119": {"description": "Benefit maximum exceeded", "fix": "Check units authorized vs billed; reauth if needed"},
    "125": {"description": "Submission/billing error", "fix": "Review claim form; resubmit with correct information"},
}

COVERAGE_POLICIES = [
    {
        "policy_id": "LCD L34521", "kind": "LCD", "title": "Anti-HER2 Therapy — Drugs & Biologicals",
        "source": "Medicare MAC · Novitas Solutions", "updated": "2026-04-01", "status": "Active",
        "summary": "Anti-HER2 monoclonal antibody therapy is covered for HER2-positive breast cancer confirmed by IHC 3+ or ISH amplification. Coverage aligns with NCCN-recommended regimens (category 1) for neoadjuvant, adjuvant, and metastatic settings.",
        "criteria": ["HER2 IHC 3+ or ISH-amplified, documented on pathology", "Stage-appropriate treatment intent",
                     "Weight-based dosing recorded", "Baseline LVEF prior to therapy initiation"],
        "covered_codes": [["J9355", "Trastuzumab"], ["J9306", "Pertuzumab"], ["J9358", "T-DXd"]],
        "icd10": ["C50.911", "C50.912", "C50.919"],
    },
    {
        "policy_id": "Policy ONC-114", "kind": "Payer policy", "title": "Adjuvant Chemotherapy — Authorization Required",
        "source": "Brightline Commercial", "updated": "2026-03-15", "status": "Active",
        "summary": "Adjuvant chemotherapy regimens require prior authorization before first administration. Authorization requests must include pathology, staging workup, and the planned regimen with J-codes and units.",
        "criteria": ["Histologic confirmation of malignancy", "Staging workup complete",
                     "Regimen matches NCCN category 1 or 2A", "Authorization obtained before date of service"],
        "covered_codes": [["96413", "Chemo infusion, first hour"], ["96415", "Each additional hour"]],
        "icd10": ["C50.411", "C18.9", "C34.10"],
    },
    {
        "policy_id": "Policy ONC-221", "kind": "Payer policy", "title": "Trastuzumab Deruxtecan — Metastatic HER2+",
        "source": "Brightline Commercial", "updated": "2026-02-20", "status": "Active",
        "summary": "T-DXd is covered for unresectable or metastatic HER2-positive breast cancer after at least one prior anti-HER2-based regimen. HER2 status must be documented within 12 months.",
        "criteria": ["HER2-positive status documented within 12 months", "At least one prior anti-HER2 regimen",
                     "Metastatic or unresectable disease", "Cardiac function assessed at baseline"],
        "covered_codes": [["J9358", "Trastuzumab deruxtecan (T-DXd)"]],
        "icd10": ["C50.911", "C50.912"],
    },
    {
        "policy_id": "Policy IO-009", "kind": "Payer policy", "title": "Pembrolizumab — Early-Stage TNBC",
        "source": "Statewide Medicaid MCO", "updated": "2026-01-30", "status": "Active",
        "summary": "Pembrolizumab is covered for high-risk early-stage triple-negative breast cancer in combination with chemotherapy as neoadjuvant treatment, then continued as a single agent adjuvant.",
        "criteria": ["Triple-negative status confirmed (ER-/PR-/HER2-)", "High-risk early-stage disease (T1c N1-2 or T2-4 N0-2)",
                     "Combination with chemotherapy in the neoadjuvant phase", "Authorization renewed every 6 months"],
        "covered_codes": [["J9271", "Pembrolizumab"]],
        "icd10": ["C50.919"],
    },
    {
        "policy_id": "NCD 90.2 / LCD L36807", "kind": "NCD", "title": "Gene Expression Profiling — Breast",
        "source": "CMS National", "updated": "2025-11-12", "status": "Active",
        "summary": "Gene expression profiling (e.g., Oncotype DX) is covered for early-stage, node-negative or limited node-positive, hormone-receptor-positive, HER2-negative breast cancer to guide adjuvant chemotherapy decisions.",
        "criteria": ["ER/PR-positive, HER2-negative disease", "Node-negative or 1-3 positive nodes",
                     "Test result will direct adjuvant chemotherapy decision", "One test per primary diagnosis"],
        "covered_codes": [["81519", "Oncotype DX Breast Recurrence Score"]],
        "icd10": ["C50.911", "C50.412"],
    },
    {
        "policy_id": "LCD L33438", "kind": "LCD", "title": "Radiation Oncology — Breast",
        "source": "Medicare MAC", "updated": "2025-12-05", "status": "Active",
        "summary": "External beam radiation therapy including IMRT is covered for breast cancer when 3D conformal planning is documented as insufficient for target coverage or normal-tissue sparing.",
        "criteria": ["Documented rationale for IMRT over 3D conformal", "Dose constraints for heart and lung recorded",
                     "Treatment plan signed prior to first fraction", "Image guidance documented per fraction"],
        "covered_codes": [["77301", "IMRT planning"], ["77385", "IMRT delivery, simple"]],
        "icd10": ["C50.911", "C50.512"],
    },
]

# Codes each policy governs — used to live-count claims citing a policy
_POLICY_CODE_INDEX = {p["policy_id"]: {c[0] for c in p["covered_codes"]} for p in COVERAGE_POLICIES}

# =========================================================================
# Rules (L0 + L1, deterministic)
# =========================================================================

def check_previsit(req: PrevisitCheckRequest) -> Tuple[Dict, List[Dict], bool]:
    elig = MOCK_ELIGIBILITY_DB.get(req.case_id, DEFAULT_ELIGIBILITY)
    soft = bool(elig.get("soft"))
    referral_required = REFERRAL_RULES.get((req.payer_id, req.plan_type), False)
    failures: List[Dict] = []
    if elig["eligibility"] != "ACTIVE":
        failures.append({"category": "ELIGIBILITY", "description": f"Eligibility is {elig['eligibility']} for the scheduled date of service"})
    if elig["network_status"] == "OUT_OF_NETWORK":
        failures.append({"category": "ELIGIBILITY", "description": "Provider is out-of-network for this payer/plan combination"})
    if elig["cob"] == "UNRESOLVED":
        failures.append({"category": "COB", "description": "Coordination of benefits is unresolved"})
    if referral_required and not req.referral_on_file:
        failures.append({"category": "REFERRAL", "description": "Required referral is missing"})
    checks = {"eligibility": elig["eligibility"], "cob": elig["cob"], "network_status": elig["network_status"],
              "referral_required": referral_required, "referral_on_file": req.referral_on_file}
    return checks, failures, soft


ICD10_REGEX = re.compile(r"^[A-TV-Z][0-9][0-9AB](\.[0-9A-Z]{1,4})?$")
NEOPLASM_PREFIXES = ("C", "D0", "D1", "D2", "D3", "D4")


def check_claim_rules(req: PrecheckRequest) -> List[Dict]:
    failures: List[Dict] = []
    seen = set()
    for jcode in req.jcodes:
        if not re.fullmatch(r"[AJQ]\d{4}", jcode):
            failures.append({"rule": "invalid_jcode_format", "category": "CPT",
                             "description": f"Drug code {jcode} is not a valid HCPCS J-code format"})
            continue
        ref = CPT_JCODE_REFERENCE.get(jcode)
        if ref and ref["auth_required"] in (True, "SOMETIMES") and not req.auth_number:
            failures.append({"rule": "auth_required_missing", "category": "AUTH",
                             "description": f"Prior authorization required and not on file for {jcode} ({ref['description']}) with {PAYER_NAMES.get(req.payer_id, req.payer_id)}"})
        if jcode in seen:
            failures.append({"rule": "duplicate_jcode", "category": "CODING",
                             "description": f"Duplicate J-code {jcode} billed on the same claim"})
        seen.add(jcode)
    # CPT format validation
    for code in req.cpt_codes:
        if not re.fullmatch(r"\d{5}", code):
            failures.append({"rule": "invalid_cpt_format", "category": "CPT",
                             "description": f"CPT code {code} is not a valid 5-digit procedure code"})
    # DX validation: format + oncology drug alignment
    if not ICD10_REGEX.fullmatch(req.primary_dx or ""):
        failures.append({"rule": "invalid_dx_format", "category": "DX",
                         "description": f"Primary diagnosis '{req.primary_dx}' is not a valid ICD-10 code"})
    elif req.jcodes and not req.primary_dx.startswith(NEOPLASM_PREFIXES):
        failures.append({"rule": "dx_drug_mismatch", "category": "DX",
                         "description": f"Primary diagnosis {req.primary_dx} is not a neoplasm (C/D) code — does not support oncology drug billing"})
    # Modifier / CPT sequencing
    cpt = set(req.cpt_codes)
    if ("96415" in cpt or "96417" in cpt) and "96413" not in cpt:
        failures.append({"rule": "modifier_sequence_invalid", "category": "MODIFIER",
                         "description": "96415/96417 billed without the required initial 96413 infusion code"})
    flat_mods = [m for row in req.modifiers for m in row if m]
    if len(flat_mods) != len(set(flat_mods)):
        failures.append({"rule": "duplicate_modifier", "category": "MODIFIER",
                         "description": "Duplicate modifier applied to the same service line"})
    if req.billed_units <= 0:
        failures.append({"rule": "invalid_units", "category": "UNITS", "description": "Billed units must be greater than zero"})
    return failures


def auth_required_for_claim(req: PrecheckRequest) -> bool:
    return any(
        (ref := CPT_JCODE_REFERENCE.get(j)) and ref["auth_required"] in (True, "SOMETIMES")
        for j in list(req.jcodes) + list(req.cpt_codes)
    )


# =========================================================================
# Heuristics (L2/L3/L4 — labeled stand-ins for trained models)
# =========================================================================

ML_HEURISTIC_VERSION = "heuristic-onc-v0.3-live"

PDL1_REGEX = re.compile(r"PD-?L1[^\d%]{0,20}(\d{1,3})\s*%")
EGFR_REGEX = re.compile(r"\bEGFR\b[^.\n]{0,40}?(mutation|mutated|positive|exon\s*\d+|L858R|del19)", re.IGNORECASE)
BRCA_REGEX = re.compile(r"\bBRCA[12]?\b[^.\n]{0,40}?(pathogenic|mutation|mutated|positive|variant)", re.IGNORECASE)
HER2_REGEX = re.compile(r"HER2[^.\n]{0,40}?(IHC\s*3\+|FISH[- ]?amplified|amplified|positive)", re.IGNORECASE)
NEGATION_REGEX = re.compile(r"(wild[- ]?type|negative|not detected|no evidence of|absent|VUS|benign variant|non[- ]?pathogenic)", re.IGNORECASE)


def _positive_finding(note: str, pattern: re.Pattern) -> bool:
    m = pattern.search(note)
    if not m:
        return False
    window = note[max(0, m.start() - 40): m.end() + 40]
    return not NEGATION_REGEX.search(window)


def extract_cdx_status(note: Optional[str]) -> Dict:
    if not note:
        return {"pdl1_present": False, "pdl1_score": None, "egfr_present": False, "brca_present": False, "her2_present": False}
    p = PDL1_REGEX.search(note)
    return {
        "pdl1_present": bool(p),
        "pdl1_score": int(p.group(1)) if p else None,
        "egfr_present": _positive_finding(note, EGFR_REGEX),
        "brca_present": _positive_finding(note, BRCA_REGEX),
        "her2_present": _positive_finding(note, HER2_REGEX),
    }


def match_nccn(jcodes: List[str], primary_dx: Optional[str]) -> Dict:
    cancer_type = CANCER_TYPE_MAP.get(primary_dx[:3]) if primary_dx else None
    for j in jcodes:
        ref = CPT_JCODE_REFERENCE.get(j)
        drug = ref["drug"] if ref else None
        if drug and drug in NCCN_TABLE:
            e = NCCN_TABLE[drug]
            return {"regimen": e["regimen"], "category": e["category"], "cancer_type": cancer_type or "/".join(e["cancer_types"])}
    return {"regimen": None, "category": None, "cancer_type": cancer_type}


def score_claim(req, rule_failures, cdx_status, nccn_match):
    reasons: List[Dict] = []
    score = 0.05
    auth_missing = any(f["rule"] == "auth_required_missing" for f in rule_failures)
    if auth_missing:
        score += 0.55
        reasons.append({"feature": "auth_status", "value": "not_obtained", "impact": "+0.55", "message": "Prior authorization not obtained"})
    else:
        score -= 0.05
        reasons.append({"feature": "auth_status", "value": "obtained_or_not_required", "impact": "-0.05", "message": "No authorization gap detected by the rules engine"})
    payer_rate = PAYER_DENIAL_RATE.get(req.payer_id, DEFAULT_PAYER_DENIAL_RATE)
    contrib = round(payer_rate * 0.4, 2)
    score += contrib
    reasons.append({"feature": "payer_denial_rate_cpt", "value": payer_rate, "impact": f"+{contrib}",
                    "message": f"{PAYER_NAMES.get(req.payer_id, req.payer_id)} denies {int(payer_rate * 100)}% of these CPT codes historically"})
    if any(cdx_status.get(k) for k in ("pdl1_present", "egfr_present", "brca_present", "her2_present")):
        score -= 0.08
        reasons.append({"feature": "cdx_test_present", "value": True, "impact": "-0.08", "message": "Required companion diagnostic result present in clinical note"})
    if nccn_match.get("category") == "1":
        score -= 0.05
        reasons.append({"feature": "nccn_category", "value": "1", "impact": "-0.05", "message": "Regimen matches NCCN Category 1 evidence"})
    other = [f for f in rule_failures if f["rule"] != "auth_required_missing"]
    for f in other:
        score += 0.15
        reasons.append({"feature": f["rule"], "value": True, "impact": "+0.15", "message": f["description"]})
    if rule_failures:
        score = max(score, 0.65)
    score = max(0.02, min(0.97, round(score, 2)))
    risk_level = "LOW" if score < 0.3 else "MEDIUM" if score < 0.6 else "HIGH" if score < 0.85 else "CRITICAL"
    if auth_missing:
        denial_class = "AUTH"
    elif other:
        denial_class = "CODING"
    elif score > 0.5:
        denial_class = "MED_NEC"
    else:
        denial_class = "CLEAN"
    auth_mass = score * 0.85 if auth_missing else 0.03
    coding_mass = score * 0.85 if other and not auth_missing else (0.1 if other else 0.02)
    med_nec_mass = max(0.02, score - auth_mass - coding_mass)
    clean_mass = max(0.0, 1 - score)
    raw = {"CLEAN": clean_mass, "AUTH": auth_mass, "MED_NEC": med_nec_mass, "CODING": coding_mass}
    total = sum(raw.values()) or 1.0
    probs = {k: round(v / total, 2) for k, v in raw.items()}
    top = sorted(reasons, key=lambda r: abs(float(str(r["impact"]).replace("+", ""))), reverse=True)[:3]
    return score, risk_level, denial_class, probs, top


APPEAL_TEMPLATE = """{today}

RE: Appeal of Claim Denial — Claim {claim_id}
Patient: {patient}
Date of Service: {dos} · Denied: {denied_date}
Billed: {amount} · Paid: {paid} · Amount in dispute: {disputed}
Denial Reason: CARC {carc} — {carc_description}{rarc_line}

To the {payer} Appeals Department:

We are writing to formally appeal the denial of claim {claim_id}.{regimen_line}

Supporting evidence on file:
{evidence_lines}

Recommended remediation per denial code {carc}: {fix}

Based on the documentation above, we request the denial be overturned and the claim reprocessed for payment of the disputed {disputed}.

Sincerely,
Billing Department
"""


def generate_appeal_letter(claim: "PrecheckResponse", carc: str, rarc: Optional[str], evidence_lines: List[str]) -> str:
    entry = CARC_REFERENCE.get(carc, {"description": "Unknown denial reason", "fix": "Review claim and resubmit with corrected information"})
    nccn = claim.nccn_match
    regimen_line = ""
    if nccn and nccn.regimen:
        regimen_line = (f" The patient was prescribed {nccn.regimen} for {nccn.cancer_type or 'the diagnosed condition'}, "
                        f"which is supported by NCCN Category {nccn.category} evidence-based guidelines.")
    lines = "\n".join(f"- {e}" for e in evidence_lines) if evidence_lines else "- No verified evidence records attached yet — attach payer evidence in the Exceptions tab to strengthen this appeal."
    return APPEAL_TEMPLATE.format(
        today=datetime.utcnow().strftime("%B %d, %Y"), claim_id=claim.claim_id,
        patient=claim.patient_name or claim.mrn or claim.claim_id,
        dos=claim.dos, denied_date=claim.denied_date or "—",
        carc=carc, carc_description=entry["description"],
        rarc_line=f" (RARC {rarc})" if rarc else "",
        payer=claim.payer_name or "Payer",
        regimen_line=regimen_line, evidence_lines=lines, fix=entry["fix"],
        amount=f"${claim.billed_amount:,.2f}",
        paid=f"${(claim.paid_amount or 0):,.2f}",
        disputed=f"${((claim.billed_amount or 0) - (claim.paid_amount or 0)):,.2f}")


def assess_appeal(claim: "PrecheckResponse", carc: str, evidence_count: int) -> Tuple[str, str]:
    """Heuristic win-likelihood + assessment text (labeled stand-in for M-06 RAG assessment)."""
    if carc in ("197", "15") and evidence_count > 0:
        return "High", (f"Administrative overturn candidate. {evidence_count} verified evidence record(s) are attached "
                        f"documenting payer contact for the authorization issue. Denials for CARC {carc} with documented "
                        "proof of authorization or timely request have the highest overturn rates.")
    if carc in ("16", "125"):
        return "Medium", ("Documentation fix. The payer identifies missing information rather than a coverage dispute — "
                          "resubmission with the requested documentation typically resolves this class of denial.")
    if carc in ("151", "119"):
        return "Low", ("Units/level dispute. The payer challenges billed units against documentation. These require "
                       "clinical records (orders, infusion logs, dosing calculations) and historically overturn less often.")
    if evidence_count > 0:
        return "Medium", (f"{evidence_count} verified evidence record(s) attached. "
                          f"CARC {carc} ({CARC_REFERENCE.get(carc, {}).get('description', 'denial')}) appeals succeed "
                          "when supporting clinical and payer documentation is included.")
    return "Medium", ("No evidence records attached yet. Attach payer verification records in the Exceptions tab — "
                      "appeals citing documented payer contact and clinical support overturn substantially more often.")


# =========================================================================
# Repo — SQLite repository. The ONLY layer that talks to storage.
# Schema is PostgreSQL-portable (TEXT/REAL/INTEGER, ISO dates, JSON payloads
# map to jsonb). Swap for Aurora by reimplementing this class with the same
# method signatures.
# =========================================================================

_SCHEMA = """
CREATE TABLE IF NOT EXISTS previsit_cases(
  case_id TEXT PRIMARY KEY,
  payload TEXT NOT NULL,
  appointment_date TEXT,
  appt_time TEXT
);
CREATE TABLE IF NOT EXISTS claims(
  claim_id TEXT PRIMARY KEY,
  payload TEXT NOT NULL,
  risk_score REAL,
  billed_amount REAL,
  payer_id TEXT,
  denied INTEGER NOT NULL DEFAULT 0,
  denial_carc TEXT,
  denial_rarc TEXT,
  denied_date TEXT
);
CREATE TABLE IF NOT EXISTS exceptions(
  exception_id TEXT PRIMARY KEY,
  case_id TEXT NOT NULL,
  category TEXT NOT NULL,
  source_layer TEXT NOT NULL,
  description TEXT NOT NULL,
  blocks_service INTEGER NOT NULL DEFAULT 1,
  owner TEXT NOT NULL DEFAULT 'Unassigned',
  due_date TEXT,
  status TEXT NOT NULL DEFAULT 'OPEN',
  resolution_notes TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS evidence(
  evidence_id TEXT PRIMARY KEY,
  exception_id TEXT NOT NULL,
  payer_source TEXT NOT NULL,
  verification_method TEXT NOT NULL,
  reference_number TEXT,
  verified_by TEXT NOT NULL,
  verified_at TEXT NOT NULL,
  attachment_url TEXT
);
CREATE TABLE IF NOT EXISTS benefits(
  patient_key TEXT PRIMARY KEY,
  deductible_total REAL, deductible_used REAL,
  oop_max REAL, oop_used REAL,
  specialist_copay REAL, coinsurance_pct REAL
);
CREATE TABLE IF NOT EXISTS users(
  username TEXT PRIMARY KEY,
  pw_salt TEXT NOT NULL,
  pw_hash TEXT NOT NULL,
  totp_secret TEXT,
  mfa_enabled INTEGER NOT NULL DEFAULT 0,
  created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS exchanges(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  direction TEXT NOT NULL,     -- outbound | inbound
  kind TEXT NOT NULL,          -- 270/271/837/835/277/ack
  partner TEXT NOT NULL,
  reference TEXT,              -- claim id, case id, trace number
  status TEXT NOT NULL,        -- sent | accepted | rejected | received | error
  detail TEXT,
  payload TEXT
);
CREATE TABLE IF NOT EXISTS settings(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sms_codes(
  username TEXT PRIMARY KEY,
  code TEXT NOT NULL,
  expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS edi_files(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  kind TEXT NOT NULL,
  raw TEXT NOT NULL,
  result TEXT NOT NULL,
  ingested_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  entity_type TEXT NOT NULL,
  entity_id TEXT NOT NULL,
  detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_exceptions_case ON exceptions(case_id);
CREATE INDEX IF NOT EXISTS idx_evidence_exc ON evidence(exception_id);
"""


class Repo:
    def __init__(self, path: str):
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            # Lightweight migrations for existing databases
            for stmt in ("ALTER TABLE users ADD COLUMN phone_country TEXT",
                         "ALTER TABLE users ADD COLUMN phone_number TEXT",
                         "ALTER TABLE users ADD COLUMN mfa_method TEXT",
                         "ALTER TABLE users ADD COLUMN role TEXT"):
                try:
                    self._conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass
            for stmt in ("ALTER TABLE claims ADD COLUMN paid_amount REAL",
                         "ALTER TABLE claims ADD COLUMN denial_reason TEXT"):
                try:
                    self._conn.execute(stmt)
                except sqlite3.OperationalError:
                    pass
            self._conn.execute("UPDATE users SET role='admin' WHERE username='admin' AND role IS NULL")
            self._conn.execute("UPDATE users SET role='billing' WHERE role IS NULL")
            self._conn.commit()

    # -- audit -------------------------------------------------------------
    def audit(self, actor: str, action: str, entity_type: str, entity_id: str, detail: str = ""):
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit_log(ts, actor, action, entity_type, entity_id, detail) VALUES (?,?,?,?,?,?)",
                (datetime.utcnow().isoformat(), actor, action, entity_type, entity_id, detail))
            self._conn.commit()

    def audit_tail(self, limit: int = 100) -> List[Dict]:
        rows = self._conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # -- previsit ----------------------------------------------------------
    def save_previsit(self, resp: PrevisitCheckResponse):
        with self._lock:
            self._conn.execute(
                "INSERT INTO previsit_cases(case_id, payload, appointment_date, appt_time) VALUES (?,?,?,?) "
                "ON CONFLICT(case_id) DO UPDATE SET payload=excluded.payload, appointment_date=excluded.appointment_date, appt_time=excluded.appt_time",
                (resp.case_id, resp.model_dump_json(), str(resp.appointment_date), resp.appt_time or ""))
            self._conn.commit()

    def list_previsit(self) -> List[PrevisitCheckResponse]:
        rows = self._conn.execute("SELECT payload FROM previsit_cases ORDER BY appointment_date, appt_time").fetchall()
        return [PrevisitCheckResponse.model_validate_json(r["payload"]) for r in rows]

    def previsit_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) c FROM previsit_cases").fetchone()["c"]

    # -- claims ------------------------------------------------------------
    def save_claim(self, resp: PrecheckResponse):
        with self._lock:
            existing = self._conn.execute(
                "SELECT denied, denial_carc, denial_rarc, denied_date, paid_amount, denial_reason FROM claims WHERE claim_id=?",
                (resp.claim_id,)).fetchone()
            if existing and existing["denied"]:
                resp.denied = True
                resp.denial_carc = existing["denial_carc"]
                resp.denial_rarc = existing["denial_rarc"]
                resp.denial_reason = existing["denial_reason"]
                resp.paid_amount = existing["paid_amount"]
                resp.denied_date = date.fromisoformat(existing["denied_date"]) if existing["denied_date"] else None
            self._conn.execute(
                "INSERT INTO claims(claim_id, payload, risk_score, billed_amount, payer_id, denied, denial_carc, denial_rarc, denied_date) "
                "VALUES (?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(claim_id) DO UPDATE SET payload=excluded.payload, risk_score=excluded.risk_score, "
                "billed_amount=excluded.billed_amount, payer_id=excluded.payer_id",
                (resp.claim_id, resp.model_dump_json(), resp.risk_score, resp.billed_amount, resp.payer_id,
                 1 if resp.denied else 0, resp.denial_carc,
                 existing["denial_rarc"] if existing else None,
                 str(resp.denied_date) if resp.denied_date else None))
            self._conn.commit()

    def get_claim(self, claim_id: str) -> Optional[PrecheckResponse]:
        row = self._conn.execute(
            "SELECT payload, denied, denial_carc, denial_rarc, denied_date, paid_amount, denial_reason FROM claims WHERE claim_id=?",
            (claim_id,)).fetchone()
        if not row:
            return None
        resp = PrecheckResponse.model_validate_json(row["payload"])
        resp.denied = bool(row["denied"])
        resp.denial_carc = row["denial_carc"]
        resp.denial_rarc = row["denial_rarc"]
        resp.paid_amount = row["paid_amount"]
        resp.denial_reason = row["denial_reason"] or (
            CARC_REFERENCE.get(row["denial_carc"] or "", {}).get("description") if row["denial_carc"] else None)
        resp.denied_date = date.fromisoformat(row["denied_date"]) if row["denied_date"] else None
        return resp

    def list_claims(self) -> List[PrecheckResponse]:
        rows = self._conn.execute("SELECT claim_id FROM claims ORDER BY risk_score DESC").fetchall()
        return [self.get_claim(r["claim_id"]) for r in rows]

    def claims_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) c FROM claims").fetchone()["c"]

    def deny_claim(self, claim_id: str, carc: str, rarc: Optional[str],
                   paid_amount: Optional[float] = None) -> Optional[PrecheckResponse]:
        with self._lock:
            row = self._conn.execute("SELECT claim_id FROM claims WHERE claim_id=?", (claim_id,)).fetchone()
            if not row:
                return None
            reason = CARC_REFERENCE.get(carc, {}).get("description", "Denial reason not in code set")
            self._conn.execute(
                "UPDATE claims SET denied=1, denial_carc=?, denial_rarc=?, denied_date=?, paid_amount=?, denial_reason=? WHERE claim_id=?",
                (carc, rarc, str(date.today()), paid_amount if paid_amount is not None else 0.0, reason, claim_id))
            self._conn.commit()
        return self.get_claim(claim_id)

    def list_denied(self) -> List[PrecheckResponse]:
        rows = self._conn.execute("SELECT claim_id FROM claims WHERE denied=1").fetchall()
        return [self.get_claim(r["claim_id"]) for r in rows]

    # -- exceptions & evidence --------------------------------------------
    def _row_to_exception(self, r: sqlite3.Row) -> ExceptionRecord:
        ev_rows = self._conn.execute("SELECT * FROM evidence WHERE exception_id=? ORDER BY verified_at", (r["exception_id"],)).fetchall()
        created = datetime.fromisoformat(r["created_at"])
        escalated = (r["owner"] == "Unassigned" and r["status"] == "OPEN"
                     and (datetime.utcnow() - created) > timedelta(hours=ESCALATION_HOURS))
        return ExceptionRecord(
            escalated=escalated,
            exception_id=r["exception_id"], case_id=r["case_id"],
            category=ExceptionCategory(r["category"]), source_layer=SourceLayer(r["source_layer"]),
            description=r["description"], blocks_service=bool(r["blocks_service"]),
            owner=r["owner"], due_date=date.fromisoformat(r["due_date"]) if r["due_date"] else None,
            status=ExceptionStatus(r["status"]), resolution_notes=r["resolution_notes"],
            created_at=datetime.fromisoformat(r["created_at"]), updated_at=datetime.fromisoformat(r["updated_at"]),
            evidence=[Evidence(evidence_id=e["evidence_id"], exception_id=e["exception_id"],
                               payer_source=e["payer_source"], verification_method=VerificationMethod(e["verification_method"]),
                               reference_number=e["reference_number"], verified_by=e["verified_by"],
                               verified_at=datetime.fromisoformat(e["verified_at"]), attachment_url=e["attachment_url"])
                      for e in ev_rows])

    def create_exception(self, case_id, category: ExceptionCategory, source_layer: SourceLayer,
                         description: str, blocks_service=True, due_date: Optional[date] = None) -> ExceptionRecord:
        # Dedup guard: re-running a check must not duplicate an open exception
        existing = self._conn.execute(
            "SELECT * FROM exceptions WHERE case_id=? AND description=? AND status IN ('OPEN','IN_PROGRESS')",
            (case_id, description)).fetchone()
        if existing:
            return self._row_to_exception(existing)
        exc = ExceptionRecord(case_id=case_id, category=category, source_layer=source_layer,
                              description=description, blocks_service=blocks_service, due_date=due_date)
        with self._lock:
            self._conn.execute(
                "INSERT INTO exceptions(exception_id, case_id, category, source_layer, description, blocks_service, owner, due_date, status, resolution_notes, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (exc.exception_id, exc.case_id, exc.category.value, exc.source_layer.value, exc.description,
                 1 if exc.blocks_service else 0, exc.owner, str(exc.due_date) if exc.due_date else None,
                 exc.status.value, exc.resolution_notes, exc.created_at.isoformat(), exc.updated_at.isoformat()))
            self._conn.commit()
        self.audit("system", "exception.created", "exception", exc.exception_id, description[:120])
        return exc

    def list_exceptions(self, case_id=None, status=None, owner=None) -> List[ExceptionRecord]:
        q = "SELECT * FROM exceptions WHERE 1=1"
        args: List[Any] = []
        if case_id: q += " AND case_id=?"; args.append(case_id)
        if status: q += " AND status=?"; args.append(status.value if hasattr(status, "value") else status)
        if owner: q += " AND LOWER(owner) LIKE ?"; args.append(f"%{owner.lower()}%")
        q += " ORDER BY COALESCE(due_date,'9999-12-31'), created_at"
        return [self._row_to_exception(r) for r in self._conn.execute(q, args).fetchall()]

    def get_exception(self, exception_id: str) -> Optional[ExceptionRecord]:
        r = self._conn.execute("SELECT * FROM exceptions WHERE exception_id=?", (exception_id,)).fetchone()
        return self._row_to_exception(r) if r else None

    def update_exception(self, exception_id: str, update: ExceptionUpdate, actor: str = "user") -> Optional[ExceptionRecord]:
        r = self._conn.execute("SELECT * FROM exceptions WHERE exception_id=?", (exception_id,)).fetchone()
        if not r:
            return None
        fields = update.model_dump(exclude_unset=True)
        sets, args = [], []
        for k, v in fields.items():
            sets.append(f"{k}=?")
            args.append(v.value if hasattr(v, "value") else (str(v) if isinstance(v, date) else v))
        sets.append("updated_at=?")
        args.append(datetime.utcnow().isoformat())
        args.append(exception_id)
        with self._lock:
            self._conn.execute(f"UPDATE exceptions SET {', '.join(sets)} WHERE exception_id=?", args)
            self._conn.commit()
        self.audit(actor, "exception.updated", "exception", exception_id, json.dumps({k: str(v) for k, v in fields.items()}))
        return self.get_exception(exception_id)

    def add_evidence(self, exception_id: str, evidence_in: EvidenceCreate, actor: str = "user") -> Optional[Evidence]:
        if not self._conn.execute("SELECT 1 FROM exceptions WHERE exception_id=?", (exception_id,)).fetchone():
            return None
        ev = Evidence(exception_id=exception_id, **evidence_in.model_dump())
        with self._lock:
            self._conn.execute(
                "INSERT INTO evidence(evidence_id, exception_id, payer_source, verification_method, reference_number, verified_by, verified_at, attachment_url) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (ev.evidence_id, ev.exception_id, ev.payer_source, ev.verification_method.value,
                 ev.reference_number, ev.verified_by, ev.verified_at.isoformat(), ev.attachment_url))
            self._conn.execute("UPDATE exceptions SET updated_at=? WHERE exception_id=?",
                               (datetime.utcnow().isoformat(), exception_id))
            self._conn.commit()
        self.audit(actor, "evidence.attached", "evidence", ev.evidence_id, f"{ev.payer_source} ref {ev.reference_number}")
        return ev

    # -- users -------------------------------------------------------------
    def get_user(self, username: str) -> Optional[Dict]:
        r = self._conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return dict(r) if r else None

    def create_user(self, username: str, password: str, role: str = "billing") -> bool:
        salt = pysecrets.token_hex(16)
        pw_hash = hashlib.sha256((salt + password).encode()).hexdigest()
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO users(username, pw_salt, pw_hash, totp_secret, mfa_enabled, created_at, role) VALUES (?,?,?,?,0,?,?)",
                (username, salt, pw_hash, None, datetime.utcnow().isoformat(), role))
            self._conn.commit()
            return cur.rowcount > 0

    def list_users(self) -> List[Dict]:
        rows = self._conn.execute("SELECT username, role, mfa_enabled, mfa_method, phone_country, phone_number, created_at FROM users ORDER BY username").fetchall()
        return [dict(r) for r in rows]

    def set_role(self, username: str, role: str):
        with self._lock:
            self._conn.execute("UPDATE users SET role=? WHERE username=?", (role, username))
            self._conn.commit()

    def set_password(self, username: str, password: str):
        salt = pysecrets.token_hex(16)
        pw_hash = hashlib.sha256((salt + password).encode()).hexdigest()
        with self._lock:
            self._conn.execute("UPDATE users SET pw_salt=?, pw_hash=? WHERE username=?", (salt, pw_hash, username))
            self._conn.commit()

    def delete_user(self, username: str):
        with self._lock:
            self._conn.execute("DELETE FROM users WHERE username=?", (username,))
            self._conn.execute("DELETE FROM sms_codes WHERE username=?", (username,))
            self._conn.commit()

    def admin_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) c FROM users WHERE role='admin'").fetchone()["c"]

    def set_totp(self, username: str, secret: Optional[str], enabled: bool):
        with self._lock:
            self._conn.execute("UPDATE users SET totp_secret=?, mfa_enabled=?, mfa_method=? WHERE username=?",
                               (secret, 1 if enabled else 0, "totp" if enabled else None, username))
            self._conn.commit()

    def set_phone(self, username: str, country: str, number: str):
        with self._lock:
            self._conn.execute("UPDATE users SET phone_country=?, phone_number=? WHERE username=?",
                               (country, number, username))
            self._conn.commit()

    def set_sms_mfa(self, username: str, enabled: bool):
        with self._lock:
            self._conn.execute("UPDATE users SET mfa_enabled=?, mfa_method=? WHERE username=?",
                               (1 if enabled else 0, "sms" if enabled else None, username))
            self._conn.commit()

    def sms_store(self, username: str, code: str, minutes: int = 5):
        with self._lock:
            self._conn.execute(
                "INSERT INTO sms_codes(username, code, expires_at) VALUES (?,?,?) "
                "ON CONFLICT(username) DO UPDATE SET code=excluded.code, expires_at=excluded.expires_at",
                (username, code, (datetime.utcnow() + timedelta(minutes=minutes)).isoformat()))
            self._conn.commit()

    def sms_peek(self, username: str) -> Optional[str]:
        r = self._conn.execute("SELECT code, expires_at FROM sms_codes WHERE username=?", (username,)).fetchone()
        if not r or datetime.fromisoformat(r["expires_at"]) < datetime.utcnow():
            return None
        return r["code"]

    def sms_check(self, username: str, code: str) -> bool:
        active = self.sms_peek(username)
        if active and hmac.compare_digest(active, (code or "").strip()):
            with self._lock:
                self._conn.execute("DELETE FROM sms_codes WHERE username=?", (username,))
                self._conn.commit()
            return True
        return False

    # -- EDI ---------------------------------------------------------------
    def save_edi(self, kind: str, raw: str, result: Dict):
        with self._lock:
            self._conn.execute("INSERT INTO edi_files(kind, raw, result, ingested_at) VALUES (?,?,?,?)",
                               (kind, raw, json.dumps(result), datetime.utcnow().isoformat()))
            self._conn.commit()

    def list_edi(self, limit: int = 20) -> List[Dict]:
        rows = self._conn.execute("SELECT id, kind, result, ingested_at FROM edi_files ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r["id"], "kind": r["kind"], "result": json.loads(r["result"]), "ingested_at": r["ingested_at"]} for r in rows]

    # -- exchanges ---------------------------------------------------------
    def log_exchange(self, direction: str, kind: str, partner: str, reference: str,
                     status: str, detail: str = "", payload: str = "") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO exchanges(ts, direction, kind, partner, reference, status, detail, payload) VALUES (?,?,?,?,?,?,?,?)",
                (datetime.utcnow().isoformat(), direction, kind, partner, reference, status, detail, payload[:20000]))
            self._conn.commit()
            return cur.lastrowid

    def list_exchanges(self, limit: int = 100) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT id, ts, direction, kind, partner, reference, status, detail FROM exchanges ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()
        return [dict(r) for r in rows]

    # -- settings ----------------------------------------------------------
    def set_setting(self, key: str, value: str):
        with self._lock:
            self._conn.execute("INSERT INTO settings(key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
            self._conn.commit()

    def get_setting(self, key: str, default: str = "") -> str:
        r = self._conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return r["value"] if r else default

    # -- benefits ----------------------------------------------------------
    def set_benefits(self, patient_key: str, b: Dict):
        with self._lock:
            self._conn.execute(
                "INSERT INTO benefits(patient_key, deductible_total, deductible_used, oop_max, oop_used, specialist_copay, coinsurance_pct) "
                "VALUES (?,?,?,?,?,?,?) ON CONFLICT(patient_key) DO UPDATE SET "
                "deductible_total=excluded.deductible_total, deductible_used=excluded.deductible_used, "
                "oop_max=excluded.oop_max, oop_used=excluded.oop_used, "
                "specialist_copay=excluded.specialist_copay, coinsurance_pct=excluded.coinsurance_pct",
                (patient_key, b.get("deductible_total"), b.get("deductible_used"), b.get("oop_max"),
                 b.get("oop_used"), b.get("specialist_copay"), b.get("coinsurance_pct")))
            self._conn.commit()

    def get_benefits(self, patient_key: str) -> Optional[Dict]:
        r = self._conn.execute("SELECT * FROM benefits WHERE patient_key=?", (patient_key,)).fetchone()
        return dict(r) if r else None


repo = Repo(DB_PATH)

# =========================================================================
# Core operations
# =========================================================================

_PREVISIT_CAT = {"ELIGIBILITY": ExceptionCategory.ELIGIBILITY, "COB": ExceptionCategory.COB, "REFERRAL": ExceptionCategory.REFERRAL}
_RULE_CAT = {"AUTH": ExceptionCategory.AUTH, "CODING": ExceptionCategory.CODING, "UNITS": ExceptionCategory.UNITS,
             "CPT": ExceptionCategory.CPT, "DX": ExceptionCategory.DX, "MODIFIER": ExceptionCategory.MODIFIER}
_ACTION_MAP = {
    "auth_required_missing": "Submit prior authorization before claim submission",
    "duplicate_jcode": "Remove duplicate J-code line before resubmission",
    "invalid_jcode_format": "Correct the drug HCPCS code (J-code) before submission",
    "invalid_cpt_format": "Correct the CPT procedure code before submission",
    "invalid_dx_format": "Correct the primary ICD-10 diagnosis code",
    "dx_drug_mismatch": "Review diagnosis sequencing — primary dx must support the billed oncology drug",
    "modifier_sequence_invalid": "Correct infusion CPT sequencing (96413 must precede 96415/96417)",
    "duplicate_modifier": "Remove the duplicated modifier from the service line",
    "invalid_units": "Correct billed units before submission",
}


def run_previsit_check(req: PrevisitCheckRequest) -> PrevisitCheckResponse:
    start = time.perf_counter()
    checks, failures, soft = check_previsit(req)
    exc_ids = []
    for f in failures:
        exc = repo.create_exception(req.case_id, _PREVISIT_CAT[f["category"]], SourceLayer.L0,
                                    f["description"], blocks_service=not soft,
                                    due_date=req.appointment_date - timedelta(days=1))
        exc_ids.append(exc.exception_id)
    status = "READY" if not failures else ("ACTION_NEEDED" if soft else "BLOCKED")
    resp = PrevisitCheckResponse(
        case_id=req.case_id, patient_ref=req.patient_ref, patient_name=req.patient_name,
        age_sex=req.age_sex, cancer=req.cancer, regimen=req.regimen, physician=req.physician,
        appt_time=req.appt_time, payer_id=req.payer_id, plan_type=req.plan_type,
        appointment_date=req.appointment_date, readiness_status=status, checks=PrevisitChecks(**checks),
        exceptions_created=exc_ids,
        next_action="; ".join(f["description"] for f in failures) if failures else "No action needed — case is ready for the scheduled visit",
        processing_ms=max(1, int((time.perf_counter() - start) * 1000)))
    repo.save_previsit(resp)
    repo.audit("system", "previsit.checked", "case", req.case_id, status)
    return resp


def run_precheck(req: PrecheckRequest) -> PrecheckResponse:
    start = time.perf_counter()
    failures = check_claim_rules(req)
    blockers = []
    for f in failures:
        exc = repo.create_exception(req.claim_id, _RULE_CAT.get(f["category"], ExceptionCategory.CODING), SourceLayer.L1,
                                    f["description"], True, due_date=req.dos - timedelta(days=1) if req.dos else None)
        blockers.append(RuleBlocker(rule=f["rule"], message=f["description"], exception_id=exc.exception_id))
    cdx = extract_cdx_status(req.clinical_note)
    nccn = match_nccn(req.jcodes, req.primary_dx)
    score, level, denial_class, probs, top = score_claim(req, failures, cdx, nccn)
    actions = [_ACTION_MAP.get(f["rule"], f["description"]) for f in failures]
    if cdx.get("pdl1_present") and nccn.get("category"):
        actions.append(f"Attach PD-L1 TPS result ({cdx.get('pdl1_score')}%) to auth request — NCCN Category {nccn['category']} for {nccn.get('cancer_type')}")
    if not actions:
        actions.append("No blocking issues detected — claim is ready for submission")
    resp = PrecheckResponse(
        claim_id=req.claim_id, patient_name=req.patient_name, mrn=req.mrn, age_sex=req.age_sex,
        stage=req.stage, line=req.line, plan_name=req.plan_name,
        payer_id=req.payer_id, payer_name=PAYER_NAMES.get(req.payer_id, req.payer_id),
        plan_type=req.plan_type, primary_dx=req.primary_dx,
        cpt_codes=req.cpt_codes, jcodes=req.jcodes, billed_amount=req.billed_amount,
        billed_units=req.billed_units, dos=req.dos,
        risk_score=score, risk_level=level, predicted_denial_class=denial_class,
        denial_probability=probs, auth_required=auth_required_for_claim(req), rule_blockers=blockers,
        shap_reasons=[ShapReason(**r) for r in top], recommended_actions=actions,
        cdx_status=CdxStatus(**cdx), nccn_match=NccnMatch(**nccn),
        processing_ms=max(1, int((time.perf_counter() - start) * 1000)), model_version=ML_HEURISTIC_VERSION)
    repo.save_claim(resp)
    repo.audit("system", "claim.prechecked", "claim", req.claim_id, f"{level} {score}")
    return resp


def _claim_evidence_lines(claim: PrecheckResponse) -> List[str]:
    lines = []
    for exc in repo.list_exceptions(case_id=claim.claim_id):
        for ev in exc.evidence:
            lines.append(f"{ev.payer_source} ref #{ev.reference_number or 'N/A'} "
                         f"(verified by {ev.verified_by} via {ev.verification_method.value} on {str(ev.verified_at)[:10]})")
    return lines


# =========================================================================
# FastAPI app
# =========================================================================

app = FastAPI(title="Precheck AI — Oncology RCM Workspace (live-data build)", version="0.4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


# ---- Authentication: password + TOTP MFA (RFC 6238) + signed sessions ----

def _totp_code(secret_b32: str, offset: int = 0, digits: int = 6, step: int = 30) -> str:
    key = base64.b32decode(secret_b32)
    counter = int(time.time()) // step + offset
    h = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    o = h[19] & 15
    code = (struct.unpack(">I", h[o:o + 4])[0] & 0x7FFFFFFF) % 10 ** digits
    return str(code).zfill(digits)


def _totp_verify(secret_b32: str, code: str) -> bool:
    code = (code or "").strip().replace(" ", "")
    return any(hmac.compare_digest(_totp_code(secret_b32, off), code) for off in (-1, 0, 1))


def _sign(payload: str) -> str:
    return hmac.new(AUTH_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]


def _make_token(username: str, kind: str = "session", minutes: int = SESSION_HOURS * 60) -> str:
    payload = f"{kind}|{username}|{int(time.time()) + minutes * 60}"
    return base64.urlsafe_b64encode(f"{payload}|{_sign(payload)}".encode()).decode()


def _read_token(token: str, expect_kind: str = "session") -> Optional[str]:
    try:
        kind, username, exp, sig = base64.urlsafe_b64decode(token.encode()).decode().split("|")
    except Exception:
        return None
    payload = f"{kind}|{username}|{exp}"
    if not hmac.compare_digest(_sign(payload), sig):
        return None
    if kind != expect_kind or int(exp) < time.time():
        return None
    return username


_AUTH_OPEN_PATHS = ("/v1/health", "/v1/auth/login", "/v1/auth/mfa/verify", "/v1/auth/mfa/demo-code",
                    "/v1/auth/mfa/resend-sms",
                    # Partner-initiated pushes authenticate by HMAC signature, not session
                    "/v1/connect/webhook")
# /jwks.json is intentionally public (Epic must fetch it) and carries only the public key


@app.middleware("http")
async def security_gate(request: Request, call_next):
    """API-key gate (PRD 9.3) + session-token gate (login/MFA)."""
    path = request.url.path
    if path.startswith("/v1/") and request.method != "OPTIONS" and path not in _AUTH_OPEN_PATHS:
        if API_KEY and request.headers.get("x-api-key") != API_KEY:
            return JSONResponse(status_code=401, content={"detail": "Missing or invalid X-API-Key"})
        if AUTH_REQUIRED:
            auth = request.headers.get("authorization", "")
            user = _read_token(auth[7:]) if auth.startswith("Bearer ") else None
            if not user:
                return JSONResponse(status_code=401, content={"detail": "Sign in required"})
            request.state.user = user
            record = repo.get_user(user)
            role = (record or {}).get("role") or "billing"
            request.state.role = role
            # RBAC (PRD 9.2 least-privilege): viewers are read-only on workspace
            # data; user administration is admin-only.
            if path.startswith("/v1/auth/users") and role != "admin":
                return JSONResponse(status_code=403, content={"detail": "Admin role required"})
            if role == "viewer" and request.method in ("POST", "PATCH", "PUT", "DELETE") \
                    and (path.startswith("/v1/oncology/") or path.startswith("/v1/fhir/")):
                return JSONResponse(status_code=403, content={"detail": "Viewer role is read-only"})
    return await call_next(request)


def _actor(x_user: Optional[str]) -> str:
    return (x_user or "").strip()[:60] or "anonymous"


class LoginRequest(BaseModel):
    username: str
    password: str


class MfaVerifyRequest(BaseModel):
    challenge: str
    code: str


class MfaCodeRequest(BaseModel):
    code: str


class SmsSetupRequest(BaseModel):
    country_code: str  # e.g. "+1", "+91"
    phone: str


def _check_password(user: Dict, password: str) -> bool:
    return hmac.compare_digest(user["pw_hash"], hashlib.sha256((user["pw_salt"] + password).encode()).hexdigest())


def _phone_hint(user: Dict) -> str:
    num = user.get("phone_number") or ""
    return f"{user.get('phone_country', '')} •••{num[-4:]}" if num else ""


def _send_sms(user: Dict, code: str):
    """SMS dispatch hook. In demo mode the code is surfaced via the demo-code
    endpoint instead of a real text. To send real SMS, plug a provider in here
    (e.g. Twilio: TWILIO_SID/TWILIO_TOKEN/TWILIO_FROM env vars) — the rest of
    the flow is provider-agnostic."""
    repo.audit("system", "auth.sms_sent", "user", user["username"],
               f"code dispatched to {_phone_hint(user)} (demo mode: shown on screen)")


def _issue_sms_code(user: Dict) -> str:
    code = str(pysecrets.randbelow(900000) + 100000)
    repo.sms_store(user["username"], code)
    _send_sms(user, code)
    return code


def _verify_mfa_code(user: Dict, code: str) -> bool:
    if (user.get("mfa_method") or "totp") == "sms":
        return repo.sms_check(user["username"], code)
    return bool(user.get("totp_secret")) and _totp_verify(user["totp_secret"], code)


@app.post("/v1/auth/login")
def auth_login(body: LoginRequest):
    user = repo.get_user(body.username.strip().lower())
    if not user or not _check_password(user, body.password):
        repo.audit(body.username.strip().lower() or "unknown", "auth.login_failed", "user", body.username, "bad credentials")
        raise HTTPException(401, "Invalid username or password")
    if user["mfa_enabled"]:
        challenge = _make_token(user["username"], kind="mfa", minutes=5)
        method = user.get("mfa_method") or "totp"
        if method == "sms":
            _issue_sms_code(user)
        repo.audit(user["username"], "auth.mfa_challenged", "user", user["username"], method)
        return {"mfa_required": True, "challenge": challenge, "method": method,
                "phone_hint": _phone_hint(user) if method == "sms" else ""}
    repo.audit(user["username"], "auth.login", "user", user["username"], "password only (MFA not enabled)")
    return {"mfa_required": False, "token": _make_token(user["username"]),
            "username": user["username"], "mfa_enabled": False}


@app.post("/v1/auth/mfa/verify")
def auth_mfa_verify(body: MfaVerifyRequest):
    username = _read_token(body.challenge, expect_kind="mfa")
    if not username:
        raise HTTPException(401, "MFA challenge expired — sign in again")
    user = repo.get_user(username)
    if not user or not _verify_mfa_code(user, body.code):
        repo.audit(username, "auth.mfa_failed", "user", username, "")
        raise HTTPException(401, "Invalid verification code")
    method = user.get("mfa_method") or "totp"
    repo.audit(username, "auth.login", "user", username, f"password + {method.upper()} MFA")
    return {"token": _make_token(username), "username": username, "mfa_enabled": True, "method": method}


@app.get("/v1/auth/me")
def auth_me(request: Request):
    username = getattr(request.state, "user", None)
    if not username:  # auth disabled — report demo identity
        return {"username": "demo", "mfa_enabled": False, "mfa_method": None, "phone_hint": "", "auth_required": AUTH_REQUIRED}
    user = repo.get_user(username) or {}
    return {"username": username, "mfa_enabled": bool(user.get("mfa_enabled")),
            "mfa_method": user.get("mfa_method"), "phone_hint": _phone_hint(user),
            "role": user.get("role") or "billing", "auth_required": AUTH_REQUIRED}


VALID_ROLES = ("admin", "billing", "viewer")


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "billing"


class UserUpdate(BaseModel):
    role: Optional[str] = None
    password: Optional[str] = None


@app.get("/v1/auth/users")
def users_list():
    return repo.list_users()


@app.post("/v1/auth/users")
def users_create(body: UserCreate, request: Request):
    username = body.username.strip().lower()
    if not re.fullmatch(r"[a-z0-9._-]{3,30}", username):
        raise HTTPException(422, "Username: 3-30 chars, letters/digits/._- only")
    if len(body.password) < 8:
        raise HTTPException(422, "Password must be at least 8 characters")
    if body.role not in VALID_ROLES:
        raise HTTPException(422, f"Role must be one of {VALID_ROLES}")
    if not repo.create_user(username, body.password, body.role):
        raise HTTPException(409, "Username already exists")
    repo.audit(getattr(request.state, "user", "admin"), "user.created", "user", username, f"role={body.role}")
    return {"username": username, "role": body.role}


@app.patch("/v1/auth/users/{username}")
def users_update(username: str, body: UserUpdate, request: Request):
    actor = getattr(request.state, "user", "admin")
    user = repo.get_user(username)
    if not user:
        raise HTTPException(404, "User not found")
    if body.role is not None:
        if body.role not in VALID_ROLES:
            raise HTTPException(422, f"Role must be one of {VALID_ROLES}")
        if user.get("role") == "admin" and body.role != "admin" and repo.admin_count() <= 1:
            raise HTTPException(409, "Cannot demote the last admin")
        repo.set_role(username, body.role)
        repo.audit(actor, "user.role_changed", "user", username, f"role={body.role}")
    if body.password is not None:
        if len(body.password) < 8:
            raise HTTPException(422, "Password must be at least 8 characters")
        repo.set_password(username, body.password)
        repo.audit(actor, "user.password_reset", "user", username, "")
    return repo.get_user(username) and {"username": username, "role": repo.get_user(username)["role"]}


@app.delete("/v1/auth/users/{username}")
def users_delete(username: str, request: Request):
    actor = getattr(request.state, "user", "admin")
    user = repo.get_user(username)
    if not user:
        raise HTTPException(404, "User not found")
    if username == actor:
        raise HTTPException(409, "You cannot delete your own account")
    if user.get("role") == "admin" and repo.admin_count() <= 1:
        raise HTTPException(409, "Cannot delete the last admin")
    repo.delete_user(username)
    repo.audit(actor, "user.deleted", "user", username, "")
    return {"deleted": username}


@app.post("/v1/auth/mfa/setup-sms")
def auth_mfa_setup_sms(body: SmsSetupRequest, request: Request):
    username = getattr(request.state, "user", "demo")
    country = body.country_code.strip()
    phone = re.sub(r"[^\d]", "", body.phone)
    if not re.fullmatch(r"\+\d{1,4}", country):
        raise HTTPException(422, "Country code must look like +1, +44, +91 …")
    if not 6 <= len(phone) <= 14:
        raise HTTPException(422, "Enter a valid phone number (6-14 digits)")
    user = repo.get_user(username)
    if not user:
        raise HTTPException(404, "User not found")
    repo.set_phone(username, country, phone)
    user = repo.get_user(username)
    _issue_sms_code(user)
    repo.audit(username, "auth.sms_setup_started", "user", username, _phone_hint(user))
    return {"phone_hint": _phone_hint(user),
            "note": "Verification code sent (demo mode: shown on screen). Confirm it to enable SMS MFA."}


@app.post("/v1/auth/mfa/enable-sms")
def auth_mfa_enable_sms(body: MfaCodeRequest, request: Request):
    username = getattr(request.state, "user", "demo")
    if not repo.sms_check(username, body.code):
        raise HTTPException(401, "Invalid or expired code")
    repo.set_sms_mfa(username, True)
    repo.audit(username, "auth.mfa_enabled", "user", username, "method=sms")
    return {"mfa_enabled": True, "method": "sms"}


@app.post("/v1/auth/mfa/setup")
def auth_mfa_setup(request: Request):
    username = getattr(request.state, "user", "demo")
    secret = base64.b32encode(pysecrets.token_bytes(20)).decode().rstrip("=")
    repo.set_totp(username, secret, enabled=False)
    repo.audit(username, "auth.mfa_setup_started", "user", username, "")
    return {"secret": secret,
            "otpauth_uri": f"otpauth://totp/PrecheckAI:{username}?secret={secret}&issuer=PrecheckAI",
            "note": "Add this secret to Google Authenticator / Authy (manual entry), then confirm with a code."}


@app.post("/v1/auth/mfa/enable")
def auth_mfa_enable(body: MfaCodeRequest, request: Request):
    username = getattr(request.state, "user", "demo")
    user = repo.get_user(username)
    if not user or not user["totp_secret"]:
        raise HTTPException(400, "Run MFA setup first")
    if not _totp_verify(user["totp_secret"], body.code):
        raise HTTPException(401, "Invalid authenticator code")
    repo.set_totp(username, user["totp_secret"], enabled=True)
    repo.audit(username, "auth.mfa_enabled", "user", username, "")
    return {"mfa_enabled": True}


@app.post("/v1/auth/mfa/disable")
def auth_mfa_disable(body: MfaCodeRequest, request: Request):
    username = getattr(request.state, "user", "demo")
    user = repo.get_user(username)
    if not user or not user["mfa_enabled"]:
        raise HTTPException(400, "MFA is not enabled")
    if not _verify_mfa_code(user, body.code):
        raise HTTPException(401, "Invalid verification code")
    if (user.get("mfa_method") or "totp") == "sms":
        repo.set_sms_mfa(username, False)
    else:
        repo.set_totp(username, None, enabled=False)
    repo.audit(username, "auth.mfa_disabled", "user", username, "")
    return {"mfa_enabled": False}


@app.post("/v1/auth/mfa/resend-sms")
def auth_mfa_resend_sms(request: Request):
    """Re-send the SMS code during login (challenge) or disable (session)."""
    auth = request.headers.get("authorization", "")
    username = _read_token(auth[7:]) if auth.startswith("Bearer ") else None
    if not username:
        username = _read_token(request.query_params.get("challenge", ""), expect_kind="mfa")
    if not username:
        raise HTTPException(401, "Sign in required")
    user = repo.get_user(username)
    if not user or not user.get("phone_number"):
        raise HTTPException(400, "No phone on file")
    _issue_sms_code(user)
    return {"sent": True, "phone_hint": _phone_hint(user)}


@app.get("/v1/auth/mfa/demo-code")
def auth_mfa_demo_code(request: Request):
    """Demo-only helper: shows the current TOTP so the flow is testable without
    an authenticator app. Returns 404 unless PRECHECK_DEMO_MFA is enabled."""
    if not DEMO_MFA_HELPER:
        raise HTTPException(404, "Not available")
    # Open path: authorize via session Bearer OR a valid MFA challenge token
    auth = request.headers.get("authorization", "")
    username = _read_token(auth[7:]) if auth.startswith("Bearer ") else None
    if not username:
        challenge = request.query_params.get("challenge", "")
        username = _read_token(challenge, expect_kind="mfa")
    if not username:
        raise HTTPException(401, "Sign in required")
    user = repo.get_user(username)
    if not user:
        raise HTTPException(400, "User not found")
    # SMS: return the active dispatched code; TOTP: return the current window code
    sms = repo.sms_peek(username)
    if sms:
        return {"code": sms, "method": "sms", "expires_in": 300,
                "warning": "Demo helper — set PRECHECK_DEMO_MFA=0 in production."}
    if user.get("totp_secret"):
        return {"code": _totp_code(user["totp_secret"]), "method": "totp",
                "expires_in": 30 - int(time.time()) % 30,
                "warning": "Demo helper — set PRECHECK_DEMO_MFA=0 in production."}
    raise HTTPException(400, "No MFA secret on file")


@app.get("/v1/health")
def health():
    return {"status": "ok", "build": "live-data + SQLite persistence",
            "db": DB_PATH,
            "model_versions": {"L0_previsit_rules": "rules-v1", "L1_claim_rules": "rules-v1",
                               "L2_risk_scoring": ML_HEURISTIC_VERSION, "L3_clinical_nlp": ML_HEURISTIC_VERSION,
                               "L4_appeal_generation": ML_HEURISTIC_VERSION}}


@app.post("/v1/oncology/previsit/check", response_model=PrevisitCheckResponse)
def previsit_check(req: PrevisitCheckRequest):
    return run_previsit_check(req)


@app.get("/v1/oncology/previsit/cases", response_model=List[PrevisitCheckResponse])
def list_previsit_cases():
    return repo.list_previsit()


@app.post("/v1/oncology/precheck", response_model=PrecheckResponse)
def precheck(req: PrecheckRequest):
    return run_precheck(req)


@app.get("/v1/oncology/precheck/claims", response_model=List[PrecheckResponse])
def list_claims():
    return repo.list_claims()


@app.get("/v1/oncology/denial/history/{claim_id}", response_model=PrecheckResponse)
def denial_history(claim_id: str):
    r = repo.get_claim(claim_id)
    if not r:
        raise HTTPException(404, "No precheck history found for this claim_id")
    return r


@app.post("/v1/oncology/claims/{claim_id}/deny", response_model=PrecheckResponse)
def deny_claim(claim_id: str, body: DenyClaimRequest, x_user: Optional[str] = Header(None)):
    if body.carc not in CARC_REFERENCE:
        raise HTTPException(422, f"Unknown CARC code {body.carc}")
    claim = repo.deny_claim(claim_id, body.carc, body.rarc, body.paid_amount)
    if not claim:
        raise HTTPException(404, "Claim not found — run precheck first")
    repo.audit(_actor(x_user), "claim.denied", "claim", claim_id,
               f"CARC {body.carc} · paid ${(body.paid_amount or 0):,.2f} · {claim.denial_reason}")
    return claim


@app.post("/v1/oncology/nlp/notes", response_model=NlpNotesResponse)
def nlp_notes(req: NlpNotesRequest):
    return NlpNotesResponse(cdx_status=CdxStatus(**extract_cdx_status(req.clinical_note)),
                            nccn_match=NccnMatch(**match_nccn(req.jcodes, req.primary_dx)),
                            model_version=ML_HEURISTIC_VERSION)


@app.post("/v1/oncology/appeal/generate", response_model=AppealGenerateResponse)
def appeal_generate(req: AppealGenerateRequest):
    claim = repo.get_claim(req.claim_id)
    if not claim:
        raise HTTPException(404, "Run /v1/oncology/precheck for this claim_id before generating an appeal")
    cited = _claim_evidence_lines(claim)
    return AppealGenerateResponse(claim_id=req.claim_id, carc=req.carc,
                                  letter_text=generate_appeal_letter(claim, req.carc, claim.denial_rarc, cited),
                                  cited_evidence=cited, model_version=ML_HEURISTIC_VERSION)


@app.get("/v1/oncology/exceptions", response_model=List[ExceptionRecord])
def list_exceptions(case_id: Optional[str] = None, status: Optional[ExceptionStatus] = None, owner: Optional[str] = None):
    return repo.list_exceptions(case_id=case_id, status=status, owner=owner)


@app.get("/v1/oncology/exceptions/{exception_id}", response_model=ExceptionRecord)
def get_exception(exception_id: str):
    exc = repo.get_exception(exception_id)
    if not exc:
        raise HTTPException(404, "Exception not found")
    return exc


@app.patch("/v1/oncology/exceptions/{exception_id}", response_model=ExceptionRecord)
def update_exception(exception_id: str, update: ExceptionUpdate, x_user: Optional[str] = Header(None)):
    exc = repo.update_exception(exception_id, update, actor=_actor(x_user))
    if not exc:
        raise HTTPException(404, "Exception not found")
    return exc


@app.post("/v1/oncology/exceptions/{exception_id}/evidence", response_model=Evidence)
def add_evidence(exception_id: str, evidence_in: EvidenceCreate, x_user: Optional[str] = Header(None)):
    ev = repo.add_evidence(exception_id, evidence_in, actor=_actor(x_user))
    if not ev:
        raise HTTPException(404, "Exception not found")
    return ev


@app.get("/v1/oncology/audit")
def audit_tail(limit: int = 100):
    return repo.audit_tail(limit)


# ---- Coverage (live claim counts) ----------------------------------------

@app.get("/v1/oncology/coverage/policies")
def coverage_policies():
    claims = repo.list_claims()
    out = []
    for p in COVERAGE_POLICIES:
        codes = _POLICY_CODE_INDEX[p["policy_id"]]
        n = sum(1 for c in claims if codes & (set(c.jcodes) | set(c.cpt_codes)))
        out.append({**p, "active_claims": n})
    return out


# ---- Eligibility (live from L0 engine + benefits) ------------------------

@app.get("/v1/oncology/eligibility/summary")
def eligibility_summary():
    cases = repo.list_previsit()
    patients = []
    for c in cases:
        st = c.checks.eligibility  # ACTIVE / INACTIVE / UNKNOWN
        status = "Active" if st == "ACTIVE" else ("Needs review" if st == "UNKNOWN" else "Inactive")
        elig_src = MOCK_ELIGIBILITY_DB.get(c.case_id, DEFAULT_ELIGIBILITY)
        benefits = repo.get_benefits(c.patient_ref)
        member_est = None
        if benefits and benefits.get("oop_max") is not None:
            ded_left = max(0.0, (benefits["deductible_total"] or 0) - (benefits["deductible_used"] or 0))
            oop_left = max(0.0, (benefits["oop_max"] or 0) - (benefits["oop_used"] or 0))
            member_est = round(min(ded_left + (benefits.get("specialist_copay") or 0), oop_left), 2)
        patients.append({
            "case_id": c.case_id, "patient_ref": c.patient_ref,
            "patient": c.patient_name or c.patient_ref, "age": c.age_sex or "—",
            "cancer": (c.cancer or "—").split("(")[0].strip(),
            "payer": PAYER_NAMES.get(c.payer_id, c.payer_id), "payer_kind": PLAN_KIND.get(c.plan_type, c.plan_type),
            "network": "OON" if c.checks.network_status == "OUT_OF_NETWORK" else ("In-Network" if c.checks.network_status == "IN_NETWORK" else "Unknown"),
            "status": status,
            "benefit": "Covered" if status == "Active" else "Unavailable",
            "notes": c.next_action if c.readiness_status != "READY" else "Eligibility confirmed",
            "readiness_status": c.readiness_status,
            "effective": elig_src.get("effective"),
            "termed": elig_src.get("termed"),
            "benefits": {**benefits, "member_est": member_est} if benefits else None,
        })
    active = sum(1 for p in patients if p["status"] == "Active")
    review = sum(1 for p in patients if p["status"] == "Needs review")
    inactive = sum(1 for p in patients if p["status"] == "Inactive")
    ready = sum(1 for p in patients if p["readiness_status"] == "READY")
    action = sum(1 for p in patients if p["readiness_status"] == "ACTION_NEEDED")
    total = len(patients) or 1
    return {
        "headline": f"{len(patients)} patients verified via L0 engine · {active} active · {inactive} inactive · {review} need review",
        "cards": [
            {"label": "Active", "value": active, "sub": "Ready to bill", "pct": f"{round(active/total*100)}% of panel", "tone": "ok"},
            {"label": "Needs review", "value": review, "sub": "Verify first", "pct": f"{round(review/total*100)}% of panel", "tone": "warn"},
            {"label": "Inactive", "value": inactive, "sub": "Coverage lapsed", "pct": f"{round(inactive/total*100)}% of panel", "tone": "bad"},
            {"label": "Action needed", "value": action, "sub": "Soft issues open", "pct": f"{round(action/total*100)}% of panel", "tone": "warn"},
        ],
        "readiness_pct": round(ready / total * 100),
        "need_action": sum(1 for p in patients if p["readiness_status"] != "READY"),
        "patients": patients,
    }


# ---- Analytics (live from repository) ------------------------------------

@app.get("/v1/oncology/analytics/summary")
def analytics_summary():
    claims = repo.list_claims()
    exceptions = repo.list_exceptions()
    total_claims = len(claims)
    clean_claims = sum(1 for c in claims if not c.rule_blockers)
    clean_rate = round(clean_claims / total_claims * 100) if total_claims else 0

    open_block_cases = {e.case_id for e in exceptions if e.blocks_service and e.status in (ExceptionStatus.OPEN, ExceptionStatus.IN_PROGRESS)}
    at_risk = sum(c.billed_amount for c in claims if c.claim_id in open_block_cases)

    resolved_block = [e for e in exceptions if e.blocks_service and e.status in (ExceptionStatus.RESOLVED, ExceptionStatus.WAIVED)]
    prevented_cases = {e.case_id for e in resolved_block}
    protected = sum(c.billed_amount for c in claims if c.claim_id in prevented_cases and c.claim_id not in open_block_cases)

    cat_counts: Dict[str, Dict[str, int]] = {}
    for e in exceptions:
        d = cat_counts.setdefault(e.category.value, {"total": 0, "open": 0})
        d["total"] += 1
        if e.status in (ExceptionStatus.OPEN, ExceptionStatus.IN_PROGRESS):
            d["open"] += 1
    trends = sorted(
        [{"label": k, "total": v["total"], "open": v["open"]} for k, v in cat_counts.items()],
        key=lambda x: -x["total"])

    payers: Dict[str, Dict] = {}
    for c in claims:
        p = payers.setdefault(c.payer_name or c.payer_id, {"claims": 0, "blockers": 0, "billed": 0.0, "risk": 0.0, "denied": 0})
        p["claims"] += 1
        p["blockers"] += len(c.rule_blockers)
        p["billed"] += c.billed_amount
        p["risk"] += c.risk_score
        p["denied"] += 1 if c.denied else 0
    payer_insights = sorted(
        [{"payer": k, "claims": v["claims"], "blockers": v["blockers"], "denied": v["denied"],
          "billed": round(v["billed"], 2), "avg_risk": round(v["risk"] / v["claims"], 2)}
         for k, v in payers.items()],
        key=lambda x: -x["billed"])

    risk_dist = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
    for c in claims:
        risk_dist[c.risk_level] = risk_dist.get(c.risk_level, 0) + 1

    return {
        "service_line": "Oncology service line · computed live from the workspace data store",
        "kpis": [
            {"label": "Claims reviewed", "value": str(total_claims), "note": "precheck runs on record"},
            {"label": "First-pass clean rate", "value": f"{clean_rate}%", "note": f"{clean_claims} of {total_claims} claims had zero hard blocks"},
            {"label": "Dollars at risk", "value": f"${at_risk:,.0f}", "note": "claims with open blocking exceptions"},
            {"label": "Denials prevented", "value": str(len(prevented_cases)), "note": "claims whose blockers were resolved pre-submission"},
            {"label": "Dollars protected", "value": f"${protected:,.0f}", "note": "billed value of prevented denials"},
        ],
        "exception_trends": trends,
        "payer_insights": payer_insights,
        "risk_distribution": risk_dist,
        "caption": "All figures computed from live claims, exceptions, and evidence in SQLite — no static demo numbers.",
    }


# ---- Appeals (live M-06 workflow: only real denied claims) ----------------

@app.get("/v1/oncology/appeals/queue")
def appeals_queue():
    denied = repo.list_denied()
    appeals = []
    for c in denied:
        carc = c.denial_carc or "16"
        entry = CARC_REFERENCE.get(carc, {"description": "Denial", "fix": ""})
        evidence_lines = _claim_evidence_lines(c)
        win, assessment = assess_appeal(c, carc, len(evidence_lines))
        denied_on = c.denied_date or date.today()
        days_left = max(0, 30 - (date.today() - denied_on).days)
        appeals.append({
            "appeal_id": f"APL-{c.claim_id}", "claim_id": c.claim_id,
            "patient": c.patient_name or c.claim_id, "amount": c.billed_amount,
            "paid_amount": c.paid_amount, "denial_reason": c.denial_reason or entry.get("description", ""),
            "rarc": c.denial_rarc,
            "at_risk": round((c.billed_amount or 0) - (c.paid_amount or 0), 2),
            "plan": c.plan_name or f"{c.payer_name} {PLAN_KIND.get(c.plan_type, '')}".strip(),
            "carc": carc, "days_left": days_left,
            "dos": str(c.dos), "denied": str(denied_on),
            "regimen": c.nccn_match.regimen or ", ".join(c.jcodes),
            "status": "Deadline near" if days_left <= 7 else ("Ready to file" if evidence_lines else "Drafting"),
            "win_likelihood": win,
            "carc_line": f"CARC {carc}" + (f" / RARC {c.denial_rarc}" if c.denial_rarc else "") + f" · {entry['description']}",
            "carc_detail": (f"Billed ${(c.billed_amount or 0):,.2f} · paid ${(c.paid_amount or 0):,.2f} · "
                            f"at risk ${((c.billed_amount or 0) - (c.paid_amount or 0)):,.2f}. "
                            f"Recommended remediation: {entry['fix']}"),
            "assessment": assessment,
            "letter": generate_appeal_letter(c, carc, c.denial_rarc, evidence_lines),
            "cited_evidence": evidence_lines,
        })
    appeals.sort(key=lambda a: a["days_left"])
    return {
        "open_count": len(appeals),
        "at_risk": round(sum(a["at_risk"] for a in appeals), 2),
        "avg_overturn_pct": 66,  # heuristic industry benchmark, labeled in UI
        "appeals": appeals,
    }


# =========================================================================
# EDI ingestion (837P claims in, 835 remittances in) — PRD Section 4.
# Simplified X12 reader: segments split on '~', elements on '*'. Production
# uses pyx12 with full 005010 implementation guides; the mapping targets and
# downstream behavior (precheck run / denial recording) are identical.
# =========================================================================

class EdiIngestRequest(BaseModel):
    raw: str


def _parse_x12_segments(raw: str) -> List[List[str]]:
    return [seg.strip().split("*") for seg in raw.replace("\n", "").split("~") if seg.strip()]


def _detect_kind(segs: List[List[str]]) -> str:
    for s in segs:
        if s[0] == "ST" and len(s) > 1:
            return {"837": "837P", "835": "835"}.get(s[1], s[1])
    return "unknown"


def ingest_837(segs: List[List[str]], actor: str) -> Dict:
    """Map key 837P loops → PrecheckRequest → run the precheck pipeline."""
    claim_id, billed, pos, dos = None, 0.0, "22", date.today()
    primary_dx, auth, payer_id, patient = "", None, "UHC01", None
    cpts, jcodes, units = [], [], 1
    for s in segs:
        tag = s[0]
        if tag == "CLM" and len(s) > 2:
            claim_id = s[1]
            try: billed = float(s[2])
            except ValueError: billed = 0.0
            if len(s) > 5 and ":" in s[5]: pos = s[5].split(":")[0]
        elif tag == "HI" and len(s) > 1 and s[1].startswith(("ABK:", "BK:")):
            primary_dx = s[1].split(":", 1)[1]
            # X12 sends ICD-10 without the dot (C3410) — reinsert it
            if len(primary_dx) > 3 and "." not in primary_dx:
                primary_dx = primary_dx[:3] + "." + primary_dx[3:]
        elif tag == "SV1" and len(s) > 1 and s[1].startswith("HC:"):
            code = s[1].split(":")[1]
            (jcodes if re.fullmatch(r"[AJQ]\d{4}", code) else cpts).append(code)
            if len(s) > 4:
                try: units = int(float(s[4]))
                except ValueError: pass
        elif tag == "REF" and len(s) > 2 and s[1] == "G1":
            auth = s[2]
        elif tag == "NM1" and len(s) > 3 and s[1] == "PR":
            name = s[3]
            payer_id = next((pid for pid, n in PAYER_NAMES.items() if n.lower() == name.lower()), name)
        elif tag == "NM1" and len(s) > 4 and s[1] == "IL":
            patient = f"{s[4]} {s[3]}".strip()
        elif tag == "DTP" and len(s) > 3 and s[1] == "472":
            try: dos = datetime.strptime(s[3], "%Y%m%d").date()
            except ValueError: pass
    if not claim_id:
        raise HTTPException(422, "No CLM segment found — not a parseable 837P")
    req = PrecheckRequest(
        claim_id=claim_id, payer_id=payer_id, plan_type="COMMERCIAL", primary_dx=primary_dx,
        cpt_codes=cpts, jcodes=jcodes, billed_units=units, billed_amount=billed,
        place_of_service=pos, dos=dos, auth_number=auth, patient_name=patient, mrn=claim_id,
        provider_npi="0000000000")
    resp = run_precheck(req)
    repo.audit(actor, "edi.837_ingested", "claim", claim_id, f"${billed:,.0f} · {len(resp.rule_blockers)} blockers")
    blockers = len(resp.rule_blockers)
    summary = (f"Claim {claim_id}"
               + (f" for {patient}" if patient else "")
               + f" · ${billed:,.2f} · {PAYER_NAMES.get(payer_id, payer_id)}"
               + f" · service date {dos}"
               + f" · {resp.risk_level} denial risk"
               + (f" · {blockers} rule check(s) failed" if blockers else " · no rule failures"))
    return {"claim_id": claim_id, "action": "precheck_run", "risk_level": resp.risk_level,
            "rule_blockers": blockers, "billed_amount": billed, "patient": patient or "",
            "payer": PAYER_NAMES.get(payer_id, payer_id), "dos": str(dos), "summary": summary}


def ingest_835(segs: List[List[str]], actor: str) -> Dict:
    """Map 835 CLP/CAS → denial labels on matching claims (the ML training signal)."""
    results = []
    current_claim = None
    for s in segs:
        tag = s[0]
        if tag == "CLP" and len(s) > 2:
            def _f(i):
                try:
                    return float(s[i]) if len(s) > i and s[i] else None
                except ValueError:
                    return None
            current_claim = {"claim_id": s[1], "status": s[2], "carc": None, "rarc": None,
                             "billed": _f(3), "paid": _f(4), "patient_resp": _f(5), "adjustment": None}
            results.append(current_claim)
        elif tag == "CAS" and current_claim is not None and len(s) > 2:
            current_claim["carc"] = current_claim["carc"] or s[2]
            if len(s) > 3 and s[3]:
                try:
                    current_claim["adjustment"] = float(s[3])
                except ValueError:
                    pass
        elif tag in ("REF", "LQ") and current_claim is not None and len(s) > 2 and s[1] in ("HI", "RARC"):
            current_claim["rarc"] = s[2]
        elif tag == "MOA" and current_claim is not None:
            rarcs = [e for e in s[2:] if e and re.fullmatch(r"[MN]\d+", e)]
            if rarcs:
                current_claim["rarc"] = rarcs[0]
    actions = []
    for r in results:
        claim = repo.get_claim(r["claim_id"])
        if not claim:
            actions.append({"claim_id": r["claim_id"], "action": "unmatched_835",
                            "detail": "No matching claim on record (logged for audit)"})
            continue
        if r["status"] in ("3", "4") or r["carc"]:  # denied / adjusted-down
            carc = r["carc"] if r["carc"] in CARC_REFERENCE else "16"
            reason = CARC_REFERENCE.get(carc, {}).get("description", "Denial reason not in code set")
            repo.deny_claim(r["claim_id"], carc, r["rarc"], r["paid"])
            repo.audit(actor, "edi.835_denial", "claim", r["claim_id"],
                       f"CARC {carc} · paid ${(r['paid'] or 0):,.2f} · {reason}")
            actions.append({"claim_id": r["claim_id"], "action": "denial_recorded", "carc": carc,
                            "rarc": r["rarc"], "reason": reason, "billed": r["billed"],
                            "paid": r["paid"], "adjustment": r["adjustment"],
                            "patient_responsibility": r["patient_resp"]})
        else:
            repo.audit(actor, "edi.835_paid", "claim", r["claim_id"], f"paid ${(r['paid'] or 0):,.2f}")
            actions.append({"claim_id": r["claim_id"], "action": "paid_clean", "carc": None,
                            "reason": None, "billed": r["billed"], "paid": r["paid"],
                            "patient_responsibility": r["patient_resp"]})
    if not results:
        raise HTTPException(422, "No CLP segments found — not a parseable 835")
    parts = []
    for a in actions:
        if a["action"] == "denial_recorded":
            money = f"billed ${(a.get('billed') or 0):,.2f}, paid ${(a.get('paid') or 0):,.2f}"
            rarc = f" / RARC {a['rarc']}" if a.get("rarc") else ""
            parts.append(f"{a['claim_id']} DENIED — CARC {a.get('carc')}{rarc}: {a.get('reason')} ({money})")
        elif a["action"] == "paid_clean":
            parts.append(f"{a['claim_id']} paid ${(a.get('paid') or 0):,.2f} of ${(a.get('billed') or 0):,.2f} billed")
        else:
            parts.append(f"{a['claim_id']} not matched to a claim on record")
    summary = f"{len(results)} claim(s) adjudicated: " + "; ".join(parts)
    return {"claims_processed": len(results), "actions": actions, "summary": summary}


# --- Human-readable X12 decoding -----------------------------------------

SEGMENT_NAMES = {
    "ISA": ("Interchange header", "Envelope start — sender/receiver and control info"),
    "GS": ("Functional group header", "Groups related transactions"),
    "ST": ("Transaction set header", "Declares the transaction type (837 claim / 835 remittance)"),
    "BHT": ("Beginning of hierarchical transaction", "Transaction purpose and reference"),
    "NM1": ("Name", "A party on the claim — payer, provider, subscriber or patient"),
    "N3": ("Address", "Street address"),
    "N4": ("Geographic location", "City, state, ZIP"),
    "CLM": ("Claim information", "Claim ID, total charge, and place of service"),
    "DTP": ("Date or time period", "Service date, admission date, etc."),
    "HI": ("Health care diagnosis codes", "ICD-10 diagnosis codes"),
    "REF": ("Reference identification", "Authorization number, referral, or other IDs"),
    "SV1": ("Professional service", "One billed service line — CPT/HCPCS, charge, units"),
    "SV2": ("Institutional service", "Institutional service line"),
    "LX": ("Service line number", "Line counter"),
    "PRV": ("Provider information", "Provider role and taxonomy"),
    "SBR": ("Subscriber information", "Payer sequence (primary/secondary) and plan"),
    "CLP": ("Claim payment information", "Adjudicated claim: status and paid amount"),
    "CAS": ("Claim adjustment", "Why money was reduced — CARC reason codes"),
    "SVC": ("Service payment information", "Per-line adjudication"),
    "BPR": ("Financial information", "Payment method and total paid"),
    "TRN": ("Trace number", "Check or EFT trace"),
    "AMT": ("Monetary amount", "Amount qualifier and value"),
    "PLB": ("Provider adjustment", "Provider-level adjustments"),
    "SE": ("Transaction set trailer", "Ends the transaction"),
    "GE": ("Functional group trailer", "Ends the group"),
    "IEA": ("Interchange trailer", "Envelope end"),
    "QTY": ("Quantity", "Units or quantity detail"),
}

CLP_STATUS = {
    "1": "Processed as primary (paid)",
    "2": "Processed as secondary (paid)",
    "3": "Processed as primary — DENIED",
    "4": "Denied",
    "19": "Processed as primary, forwarded to additional payer",
    "22": "Reversal of previous payment",
    "25": "Predetermination pricing only",
}

POS_NAMES = {"11": "Office", "19": "Off-campus outpatient hospital", "21": "Inpatient hospital",
             "22": "On-campus outpatient hospital", "23": "Emergency room", "24": "Ambulatory surgical center",
             "49": "Independent clinic", "81": "Independent laboratory"}


def _code_label(code: str) -> str:
    ref = CPT_JCODE_REFERENCE.get(code)
    return ref["description"] if ref else "—"


def decode_x12(raw: str) -> Dict:
    """Turn a raw X12 string into labeled segments + a plain-English summary."""
    segs = _parse_x12_segments(raw)
    kind = _detect_kind(segs)
    decoded = []
    for s in segs:
        tag = s[0]
        name, desc = SEGMENT_NAMES.get(tag, (tag, "Segment"))
        detail = ""
        if tag == "ST" and len(s) > 1:
            detail = {"837": "837 — Health care claim", "835": "835 — Claim payment/advice"}.get(s[1], s[1])
        elif tag == "CLM" and len(s) > 2:
            pos = s[5].split(":")[0] if len(s) > 5 and ":" in s[5] else ""
            detail = f"Claim {s[1]} · total ${float(s[2]):,.2f}" + (f" · place of service {pos} ({POS_NAMES.get(pos, 'unknown')})" if pos else "")
        elif tag == "NM1" and len(s) > 3:
            role = {"PR": "Payer", "IL": "Subscriber/patient", "85": "Billing provider",
                    "82": "Rendering provider", "QC": "Patient", "40": "Receiver", "41": "Submitter"}.get(s[1], s[1])
            who = " ".join(x for x in [s[4] if len(s) > 4 else "", s[3]] if x)
            detail = f"{role}: {who.strip()}"
        elif tag == "HI" and len(s) > 1:
            codes = []
            for el in s[1:]:
                if ":" in el:
                    q, c = el.split(":", 1)
                    if len(c) > 3 and "." not in c:
                        c = c[:3] + "." + c[3:]
                    codes.append(c)
            detail = "Diagnosis: " + ", ".join(codes) if codes else ""
        elif tag == "SV1" and len(s) > 1 and ":" in s[1]:
            code = s[1].split(":")[1]
            charge = f"${float(s[2]):,.2f}" if len(s) > 2 and s[2] else ""
            units = s[4] if len(s) > 4 else ""
            detail = f"{code} ({_code_label(code)}) · {charge}" + (f" · {units} unit(s)" if units else "")
        elif tag == "DTP" and len(s) > 3:
            qual = {"472": "Service date", "434": "Statement period", "096": "Discharge"}.get(s[1], s[1])
            d = s[3]
            pretty = f"{d[0:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 and d.isdigit() else d
            detail = f"{qual}: {pretty}"
        elif tag == "REF" and len(s) > 2:
            qual = {"G1": "Prior authorization number", "9F": "Referral number", "EA": "Medical record number"}.get(s[1], s[1])
            detail = f"{qual}: {s[2] or '(empty)'}"
        elif tag == "CLP" and len(s) > 2:
            status = CLP_STATUS.get(s[2], f"status {s[2]}")
            billed = f"${float(s[3]):,.2f}" if len(s) > 3 and s[3] else ""
            paid = f"${float(s[4]):,.2f}" if len(s) > 4 and s[4] else ""
            detail = f"Claim {s[1]} · {status}" + (f" · billed {billed}" if billed else "") + (f" · paid {paid}" if paid else "")
        elif tag == "CAS" and len(s) > 2:
            group = {"CO": "Contractual obligation", "PR": "Patient responsibility",
                     "OA": "Other adjustment", "PI": "Payer initiated"}.get(s[1], s[1])
            carc = s[2]
            amt = f"${float(s[3]):,.2f}" if len(s) > 3 and s[3] else ""
            reason = CARC_REFERENCE.get(carc, {}).get("description", "Unknown reason code")
            detail = f"{group} · CARC {carc} — {reason}" + (f" · {amt}" if amt else "")
        decoded.append({"tag": tag, "name": name, "description": desc,
                        "detail": detail, "raw": "*".join(s)})

    # Plain-English summary of the whole transaction
    summary: List[str] = []
    if kind == "837P":
        clm = next((s for s in segs if s[0] == "CLM"), None)
        pat = next((s for s in segs if s[0] == "NM1" and len(s) > 1 and s[1] == "IL"), None)
        payer = next((s for s in segs if s[0] == "NM1" and len(s) > 1 and s[1] == "PR"), None)
        lines = [s for s in segs if s[0] == "SV1"]
        dx = next((d["detail"] for d in decoded if d["tag"] == "HI" and d["detail"]), "")
        auth = next((s[2] for s in segs if s[0] == "REF" and len(s) > 2 and s[1] == "G1"), "")
        if pat:
            summary.append(f"Patient: {' '.join(x for x in [pat[4] if len(pat) > 4 else '', pat[3]] if x).strip()}")
        if payer:
            summary.append(f"Payer: {payer[3]}")
        if clm:
            summary.append(f"Claim {clm[1]} for ${float(clm[2]):,.2f}")
        if dx:
            summary.append(dx)
        summary.append(f"{len(lines)} service line(s): " + ", ".join(
            f"{s[1].split(':')[1]} ({_code_label(s[1].split(':')[1])})" for s in lines if ":" in s[1]))
        summary.append(f"Prior authorization: {auth}" if auth else "Prior authorization: none on file")
    elif kind == "835":
        for s in segs:
            if s[0] == "CLP" and len(s) > 2:
                summary.append(f"Claim {s[1]}: {CLP_STATUS.get(s[2], 'status ' + s[2])}")
            if s[0] == "CAS" and len(s) > 2:
                summary.append(f"Adjustment CARC {s[2]} — {CARC_REFERENCE.get(s[2], {}).get('description', 'unknown reason')}")
    return {"kind": kind, "segments": decoded, "summary": summary}


class EdiDecodeRequest(BaseModel):
    raw: str


@app.post("/v1/oncology/edi/decode")
def edi_decode(body: EdiDecodeRequest):
    """Read-only: decode an X12 transaction for review before ingesting."""
    return decode_x12(body.raw)


SAMPLE_837 = ("ISA*00*~ST*837*0001~"
              "NM1*IL*1*OKONKWO*ADAEZE~NM1*PR*2*BCBS~"
              "CLM*CLM001220*31500***22:B:1~DTP*472*D8*20260715~"
              "HI*ABK:C50911~REF*G1*~"
              "SV1*HC:96413*9500*UN*1~SV1*HC:96415*4200*UN*2~SV1*HC:J9355*17800*UN*40~"
              "SE*10*0001~")

SAMPLE_835 = ("ISA*00*~ST*835*0002~"
              "CLP*CLM001220*3*31500*0~CAS*CO*197*31500~"
              "SE*4*0002~")


@app.post("/v1/oncology/edi/ingest")
def edi_ingest(body: EdiIngestRequest, x_user: Optional[str] = Header(None)):
    segs = _parse_x12_segments(body.raw)
    kind = _detect_kind(segs)
    actor = _actor(x_user)
    if kind == "837P":
        result = ingest_837(segs, actor)
    elif kind == "835":
        result = ingest_835(segs, actor)
    else:
        raise HTTPException(422, f"Unrecognized transaction type '{kind}' — expected ST*837 or ST*835")
    repo.save_edi(kind, body.raw, result)
    return {"kind": kind, "result": result}


class EdiBatchFile(BaseModel):
    name: str
    raw: str


class EdiBatchRequest(BaseModel):
    files: List[EdiBatchFile]


@app.post("/v1/oncology/edi/ingest-batch")
def edi_ingest_batch(body: EdiBatchRequest, x_user: Optional[str] = Header(None)):
    if not body.files:
        raise HTTPException(422, "No files provided")
    if len(body.files) > 50:
        raise HTTPException(422, "Max 50 files per batch")
    actor = _actor(x_user)
    results = []
    for f in body.files:
        try:
            segs = _parse_x12_segments(f.raw)
            kind = _detect_kind(segs)
            if kind == "837P":
                result = ingest_837(segs, actor)
            elif kind == "835":
                result = ingest_835(segs, actor)
            else:
                raise HTTPException(422, f"Unrecognized transaction type '{kind}'")
            repo.save_edi(kind, f.raw, result)
            results.append({"name": f.name, "ok": True, "kind": kind, "result": result})
        except HTTPException as e:
            results.append({"name": f.name, "ok": False, "kind": None, "error": e.detail})
        except Exception as e:
            results.append({"name": f.name, "ok": False, "kind": None, "error": str(e)[:200]})
    ok_n = sum(1 for r in results if r["ok"])
    repo.audit(actor, "edi.batch_ingested", "edi", f"{len(body.files)} files", f"{ok_n} ok, {len(results) - ok_n} failed")
    return {"total": len(results), "succeeded": ok_n, "failed": len(results) - ok_n, "results": results}


@app.get("/v1/oncology/edi/files")
def edi_files(limit: int = 20):
    return repo.list_edi(limit)


@app.get("/v1/oncology/edi/samples")
def edi_samples():
    return {"sample_837": SAMPLE_837, "sample_835": SAMPLE_835}


# =========================================================================
# Real-time exchange layer — clearinghouse / EHR connectivity.
#
# Transactions: 270/271 eligibility, 837 claim submission, 277 status,
# 835 remittance (pulled or pushed by webhook).
#
# Adapters isolate the transport so partners plug in without touching the
# workflow:
#   mock  — built-in simulator, works offline (demo default)
#   rest  — JSON/X12-over-HTTPS API (Availity, Change/Optum, Waystar style)
#   sftp  — batch drop/pickup (documented; enable with a partner's creds)
# Inbound pushes arrive at /v1/connect/webhook and are HMAC-verified.
# =========================================================================

PARTNER_PRESETS = [
    {"id": "mock", "label": "Built-in simulator (offline demo)", "mode": "mock", "base_url": ""},
    {"id": "availity", "label": "Availity Essentials (REST)", "mode": "rest", "base_url": "https://api.availity.com"},
    {"id": "optum", "label": "Optum / Change Healthcare (REST)", "mode": "rest", "base_url": "https://apis.changehealthcare.com"},
    {"id": "waystar", "label": "Waystar (REST)", "mode": "rest", "base_url": "https://api.waystar.com"},
    {"id": "custom", "label": "Custom partner endpoint", "mode": "rest", "base_url": ""},
]


def _partner_cfg() -> Dict:
    return {
        "partner_id": repo.get_setting("ch_partner", "mock"),
        "mode": repo.get_setting("ch_mode", "mock"),
        "base_url": repo.get_setting("ch_base_url", ""),
        "trading_partner_id": repo.get_setting("ch_tpid", "DEMO-TP-001"),
        "has_api_key": bool(repo.get_setting("ch_api_key", "")),
        "webhook_secret": repo.get_setting("ch_webhook_secret", ""),
    }


def _partner_post(path: str, body: Dict) -> Dict:
    """REST adapter. Sends JSON, expects JSON. X12 payloads ride in the body."""
    cfg = _partner_cfg()
    if not cfg["base_url"]:
        raise HTTPException(400, "No partner base URL configured")
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {repo.get_setting('ch_api_key', '')}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise HTTPException(502, f"Partner returned {e.code}: {e.read().decode()[:200]}")
    except Exception as e:
        raise HTTPException(502, f"Could not reach partner: {str(e)[:150]}")


# ---- 270/271 real-time eligibility ---------------------------------------

def build_270(case: PrevisitCheckResponse) -> str:
    tp = _partner_cfg()["trading_partner_id"]
    return (f"ISA*00*~GS*HS*NETGAIN*{tp}*~ST*270*0001~BHT*0022*13*{case.case_id}*~"
            f"HL*1**20*1~NM1*PR*2*{PAYER_NAMES.get(case.payer_id, case.payer_id)}*****PI*{case.payer_id}~"
            f"HL*2*1*21*1~NM1*1P*2*NETGAIN ONCOLOGY*****XX*{'0000000000'}~"
            f"HL*3*2*22*0~NM1*IL*1*{(case.patient_name or case.patient_ref).upper()}~"
            f"DMG*D8*19700101~DTP*291*D8*{str(case.appointment_date).replace('-', '')}~"
            f"EQ*30~SE*12*0001~")


def mock_271(case: PrevisitCheckResponse) -> Dict:
    """Simulated payer response: active coverage with benefit accumulators."""
    seed = sum(ord(c) for c in case.case_id)
    ded_total = 1500 + (seed % 4) * 500
    return {
        "eligibility": "ACTIVE" if case.checks.eligibility == "ACTIVE" else case.checks.eligibility,
        "network_status": case.checks.network_status,
        "cob": case.checks.cob,
        "benefits": {
            "deductible_total": ded_total,
            "deductible_used": min(ded_total, (seed % 7) * 250),
            "oop_max": 6000 + (seed % 3) * 1000,
            "oop_used": (seed % 9) * 400,
            "specialist_copay": 35 + (seed % 4) * 5,
            "coinsurance_pct": 10 + (seed % 3) * 5,
        },
        "trace": f"MOCK271-{seed}",
    }


class EligibilityCheckRequest(BaseModel):
    case_id: str


@app.post("/v1/connect/eligibility")
def connect_eligibility(body: EligibilityCheckRequest, x_user: Optional[str] = Header(None)):
    """Real-time 270 → 271 round trip; writes benefits back onto the case."""
    case = next((c for c in repo.list_previsit() if c.case_id == body.case_id), None)
    if not case:
        raise HTTPException(404, "Case not found")
    cfg = _partner_cfg()
    x12_270 = build_270(case)
    repo.log_exchange("outbound", "270", cfg["partner_id"], case.case_id, "sent",
                      f"Eligibility inquiry for {case.patient_name or case.patient_ref}", x12_270)
    if cfg["mode"] == "mock":
        resp = mock_271(case)
    else:
        resp = _partner_post("/eligibility/v1", {"x12": x12_270, "tradingPartnerId": cfg["trading_partner_id"]})
    benefits = resp.get("benefits") or {}
    if benefits:
        repo.set_benefits(case.patient_ref, benefits)
    repo.log_exchange("inbound", "271", cfg["partner_id"], case.case_id, "received",
                      f"Coverage {resp.get('eligibility', '?')} · trace {resp.get('trace', '')}", json.dumps(resp))
    repo.audit(_actor(x_user), "connect.eligibility_checked", "case", case.case_id, cfg["partner_id"])
    return {"case_id": case.case_id, "response": resp, "benefits_updated": bool(benefits)}


# ---- 837 claim submission -------------------------------------------------

def build_837(claim: PrecheckResponse) -> str:
    tp = _partner_cfg()["trading_partner_id"]
    lines = "".join(f"SV1*HC:{c}*0*UN*{claim.billed_units}~" for c in (claim.cpt_codes + claim.jcodes))
    return (f"ISA*00*~GS*HC*NETGAIN*{tp}*~ST*837*0001~"
            f"NM1*IL*1*{(claim.patient_name or claim.claim_id).upper()}~"
            f"NM1*PR*2*{claim.payer_name}~"
            f"CLM*{claim.claim_id}*{claim.billed_amount:.2f}***{claim.plan_type[:2]}:B:1~"
            f"DTP*472*D8*{str(claim.dos).replace('-', '')}~HI*ABK:{claim.primary_dx.replace('.', '')}~"
            f"{lines}SE*10*0001~")


class SubmitClaimRequest(BaseModel):
    claim_id: str
    force: bool = False  # submit even when rule blockers are open


@app.post("/v1/connect/submit")
def connect_submit(body: SubmitClaimRequest, x_user: Optional[str] = Header(None)):
    """Submit an 837 — refuses claims with open hard blockers unless forced."""
    claim = repo.get_claim(body.claim_id)
    if not claim:
        raise HTTPException(404, "Claim not found")
    open_blockers = [e for e in repo.list_exceptions(case_id=claim.claim_id)
                     if e.blocks_service and e.status in (ExceptionStatus.OPEN, ExceptionStatus.IN_PROGRESS)]
    if open_blockers and not body.force:
        raise HTTPException(409, f"{len(open_blockers)} open blocking exception(s) — resolve them or resubmit with force=true")
    cfg = _partner_cfg()
    x12_837 = build_837(claim)
    if cfg["mode"] == "mock":
        ack = {"status": "accepted", "trace": f"MOCK837-{claim.claim_id}",
               "message": "Claim accepted by clearinghouse for payer forwarding"}
    else:
        ack = _partner_post("/claims/v1/submission", {"x12": x12_837, "tradingPartnerId": cfg["trading_partner_id"]})
    repo.log_exchange("outbound", "837", cfg["partner_id"], claim.claim_id,
                      ack.get("status", "sent"),
                      f"${claim.billed_amount:,.2f} to {claim.payer_name} · trace {ack.get('trace', '')}", x12_837)
    repo.audit(_actor(x_user), "connect.claim_submitted", "claim", claim.claim_id,
               f"{cfg['partner_id']} · {ack.get('status')}" + (" · FORCED" if open_blockers else ""))
    return {"claim_id": claim.claim_id, "acknowledgement": ack,
            "submitted_with_open_blockers": bool(open_blockers)}


# ---- 835 remittance retrieval --------------------------------------------

@app.post("/v1/connect/poll-remits")
def connect_poll_remits(x_user: Optional[str] = Header(None)):
    """Pull available 835s and run them through the same ingestion path."""
    cfg = _partner_cfg()
    actor = _actor(x_user)
    if cfg["mode"] == "mock":
        submitted = [e for e in repo.list_exchanges(200) if e["kind"] == "837"]
        remits = []
        for e in submitted[:3]:
            claim = repo.get_claim(e["reference"])
            if not claim or claim.denied:
                continue
            # Simulate the payer's adjudication using the precheck signal
            denied = bool(claim.rule_blockers)
            carc = "197" if denied and claim.predicted_denial_class == "AUTH" else ("16" if denied else "")
            paid = 0.0 if denied else round(claim.billed_amount * 0.82, 2)
            status = "3" if denied else "1"
            remits.append(f"ISA*00*~ST*835*0002~CLP*{claim.claim_id}*{status}*{claim.billed_amount:.2f}*{paid:.2f}~"
                          + (f"CAS*CO*{carc}*{claim.billed_amount:.2f}~" if denied else "")
                          + "SE*4*0002~")
    else:
        resp = _partner_post("/remits/v1/list", {"tradingPartnerId": cfg["trading_partner_id"]})
        remits = resp.get("files", [])
    results = []
    for raw in remits:
        segs = _parse_x12_segments(raw)
        try:
            out = ingest_835(segs, actor)
            repo.save_edi("835", raw, out)
            repo.log_exchange("inbound", "835", cfg["partner_id"],
                              out["actions"][0]["claim_id"] if out.get("actions") else "",
                              "received", out.get("summary", ""), raw)
            results.append(out)
        except HTTPException as e:
            repo.log_exchange("inbound", "835", cfg["partner_id"], "", "error", str(e.detail)[:200], raw)
    repo.audit(actor, "connect.remits_polled", "edi", cfg["partner_id"], f"{len(results)} remit(s)")
    return {"remits_processed": len(results), "results": results}


# ---- Inbound webhook (partner-initiated, real time) ----------------------

class WebhookPayload(BaseModel):
    kind: str          # 835 | 277 | 271
    x12: Optional[str] = None
    reference: Optional[str] = None


@app.post("/v1/connect/webhook")
async def connect_webhook(request: Request):
    """Partners POST here for real-time pushes. Verified with an HMAC signature
    (X-Signature: sha256=<hex>) over the raw body using the webhook secret."""
    secret = repo.get_setting("ch_webhook_secret", "")
    raw_body = await request.body()
    if secret:
        sig = (request.headers.get("x-signature") or "").replace("sha256=", "")
        expected = hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            repo.log_exchange("inbound", "webhook", _partner_cfg()["partner_id"], "", "rejected",
                              "Invalid or missing HMAC signature", raw_body.decode(errors="replace")[:500])
            raise HTTPException(401, "Invalid signature")
    try:
        body = WebhookPayload.model_validate_json(raw_body)
    except Exception:
        raise HTTPException(422, "Body must be {kind, x12, reference}")
    partner = _partner_cfg()["partner_id"]
    if body.kind == "835" and body.x12:
        out = ingest_835(_parse_x12_segments(body.x12), "webhook")
        repo.save_edi("835", body.x12, out)
        repo.log_exchange("inbound", "835", partner, body.reference or "", "received",
                          out.get("summary", ""), body.x12)
        return {"accepted": True, "result": out}
    repo.log_exchange("inbound", body.kind, partner, body.reference or "", "received",
                      "Status/acknowledgement received", body.x12 or "")
    return {"accepted": True}


# ---- Config + log --------------------------------------------------------

class PartnerConfigRequest(BaseModel):
    partner_id: str
    mode: str = "mock"
    base_url: str = ""
    api_key: Optional[str] = None
    trading_partner_id: str = "DEMO-TP-001"


@app.get("/v1/connect/config")
def connect_config(request: Request):
    cfg = _partner_cfg()
    if not cfg["webhook_secret"]:
        repo.set_setting("ch_webhook_secret", pysecrets.token_hex(16))
        cfg = _partner_cfg()
    base = str(request.base_url).rstrip("/")
    return {**cfg, "presets": PARTNER_PRESETS, "webhook_url": f"{base}/v1/connect/webhook",
            "webhook_public": not any(h in base for h in ("localhost", "127.0.0.1"))}


@app.post("/v1/connect/config")
def connect_config_set(body: PartnerConfigRequest, request: Request, x_user: Optional[str] = Header(None)):
    if body.mode not in ("mock", "rest", "sftp"):
        raise HTTPException(422, "mode must be mock, rest or sftp")
    repo.set_setting("ch_partner", body.partner_id)
    repo.set_setting("ch_mode", body.mode)
    repo.set_setting("ch_base_url", body.base_url.strip())
    repo.set_setting("ch_tpid", body.trading_partner_id.strip() or "DEMO-TP-001")
    if body.api_key is not None:
        repo.set_setting("ch_api_key", body.api_key.strip())
    repo.audit(_actor(x_user), "connect.config_changed", "partner", body.partner_id, body.mode)
    return connect_config(request)


@app.get("/v1/connect/log")
def connect_log(limit: int = 100):
    return repo.list_exchanges(limit)


# =========================================================================
# Validation framework — rules engine vs expert manual review (roadmap #4).
# Expert labels are the gold standard; the run computes precision/recall/F1,
# missed blockers (false negatives), and false flags (false positives).
# =========================================================================

def _vs(claim_id, description, expert, **kw):
    defaults = dict(claim_id=claim_id, payer_id="UHC01", plan_type="COMMERCIAL", primary_dx="C34.10",
                    cpt_codes=["96413"], jcodes=[], billed_units=10, billed_amount=5000.0,
                    place_of_service="22", dos=date(2026, 7, 1), auth_number="AUTH-1",
                    provider_npi="0000000000")
    defaults.update(kw)
    return {"scenario_id": claim_id, "description": description,
            "expert_blockers": expert, "request": PrecheckRequest(**defaults)}


VALIDATION_SET = [
    _vs("VAL-01", "Clean claim, auth on file", []),
    _vs("VAL-02", "Pembrolizumab without prior auth", ["auth_required_missing"],
        jcodes=["J9271"], auth_number=None),
    _vs("VAL-03", "Trastuzumab with valid auth", [], jcodes=["J9355"], primary_dx="C50.911"),
    _vs("VAL-04", "96415 billed without initial 96413", ["modifier_sequence_invalid"],
        cpt_codes=["96415"]),
    _vs("VAL-05", "Invalid CPT format on service line", ["invalid_cpt_format"],
        cpt_codes=["96413", "ABC12"]),
    _vs("VAL-06", "Non-neoplasm dx on drug claim", ["dx_drug_mismatch"],
        jcodes=["J9045"], primary_dx="R51.9"),
    _vs("VAL-07", "Malformed ICD-10 primary dx", ["invalid_dx_format"], primary_dx="CAT34"),
    _vs("VAL-08", "Duplicate J-code lines", ["duplicate_jcode", "auth_required_missing"],
        jcodes=["J9271", "J9271"], auth_number=None),
    _vs("VAL-09", "Zero billed units", ["invalid_units"], billed_units=0),
    _vs("VAL-10", "Duplicate modifier on line", ["duplicate_modifier"],
        modifiers=[["JW", "JW", None, None]]),
    # Designed disagreements — measure honest gaps:
    _vs("VAL-11", "Expert flags thin clinical documentation (not rule-detectable today)",
        ["clinical_doc_insufficient"], jcodes=["J9055"], primary_dx="C18.9"),
    _vs("VAL-12", "Payer waives auth for 96413 on this plan (expert: no auth needed)",
        [], cpt_codes=["96413"], jcodes=[], auth_number=None),
]


@app.post("/v1/oncology/validation/run")
def validation_run(x_user: Optional[str] = Header(None)):
    tp = fp = fn = 0
    details, missed, false_flags = [], [], []
    for sc in VALIDATION_SET:
        engine = {f["rule"] for f in check_claim_rules(sc["request"])}
        expert = set(sc["expert_blockers"])
        s_tp = engine & expert
        s_fp = engine - expert
        s_fn = expert - engine
        tp += len(s_tp); fp += len(s_fp); fn += len(s_fn)
        for r in s_fn:
            missed.append({"scenario": sc["scenario_id"], "rule": r, "description": sc["description"]})
        for r in s_fp:
            false_flags.append({"scenario": sc["scenario_id"], "rule": r, "description": sc["description"]})
        details.append({"scenario_id": sc["scenario_id"], "description": sc["description"],
                        "expert": sorted(expert), "engine": sorted(engine),
                        "match": engine == expert})
    precision = round(tp / (tp + fp), 3) if (tp + fp) else 1.0
    recall = round(tp / (tp + fn), 3) if (tp + fn) else 1.0
    f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) else 0.0
    repo.audit(_actor(x_user), "validation.run", "validation", f"n={len(VALIDATION_SET)}",
               f"P={precision} R={recall} F1={f1}")
    return {
        "scenarios": len(VALIDATION_SET),
        "exact_match": sum(1 for d in details if d["match"]),
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "missed_blockers": missed, "false_flags": false_flags, "details": details,
        "note": "Expert labels are the gold standard. Validation runs the L1 rules engine only — no data is written to the workspace.",
    }


# =========================================================================
# FHIR R4 connector (PRD Section 8 — Epic/EHR integration, milestone M9).
# Works against any FHIR R4 base URL: SMART Health IT's open sandbox by
# default (synthetic patients, no auth), or Epic's sandbox with an access
# token obtained from a registered app at fhir.epic.com. Production Epic
# requires App Orchard registration + SMART OAuth — this connector is the
# same resource layer (Patient / Condition / Coverage) that flow uses.
# =========================================================================

FHIR_PRESETS = [
    {"id": "smart", "label": "SMART Health IT sandbox (open, no auth)", "base_url": "https://r4.smarthealthit.org"},
    {"id": "hapi", "label": "HAPI FHIR public test server (open)", "base_url": "https://hapi.fhir.org/baseR4"},
    {"id": "epic", "label": "Epic sandbox (register free at fhir.epic.com, paste token)", "base_url": "https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4"},
]
DEFAULT_FHIR_BASE = FHIR_PRESETS[0]["base_url"]


class FhirConfigRequest(BaseModel):
    base_url: str
    token: Optional[str] = None


# ---- Epic backend OAuth (JWT client credentials, SMART Backend Services) ---
EPIC_TOKEN_URL = "https://fhir.epic.com/interconnect-fhir-oauth/oauth2/token"
EPIC_R4_BASE = "https://fhir.epic.com/interconnect-fhir-oauth/api/FHIR/R4"
KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epic_private.pem")


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _load_or_create_key():
    """RSA key used to sign Epic client-assertion JWTs.

    Resolution order:
      1. PRECHECK_EPIC_PRIVATE_KEY env var (PEM text) — use this in hosted
         environments so the key survives redeploys and keeps matching the
         JWK Set already published to Epic.
      2. epic_private.pem next to this file.
      3. Generate a new key (first local run).
    """
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        raise HTTPException(500, "The 'cryptography' package is required for Epic auth — pip install cryptography")
    env_pem = os.environ.get("PRECHECK_EPIC_PRIVATE_KEY", "").strip()
    if env_pem:
        return serialization.load_pem_private_key(env_pem.replace("\\n", "\n").encode(), password=None)
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with open(KEY_PATH, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM,
                                  serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    return key


def _jwk() -> Dict:
    key = _load_or_create_key()
    pub = key.public_key().public_numbers()
    n = _b64url(pub.n.to_bytes((pub.n.bit_length() + 7) // 8, "big"))
    e = _b64url(pub.e.to_bytes((pub.e.bit_length() + 7) // 8, "big"))
    thumb = json.dumps({"e": e, "kty": "RSA", "n": n}, separators=(",", ":"), sort_keys=True)
    kid = _b64url(hashlib.sha256(thumb.encode()).digest())
    return {"kty": "RSA", "n": n, "e": e, "kid": kid, "use": "sig", "alg": "RS384"}


@app.get("/jwks.json")
def jwks_document():
    """Public JWK Set — paste this endpoint's URL into Epic's 'JWK Set URL' field.
    Deliberately unauthenticated: it contains only the PUBLIC key."""
    return {"keys": [_jwk()]}


def _epic_client_assertion(client_id: str) -> str:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    key = _load_or_create_key()
    now = int(time.time())
    header = {"alg": "RS384", "typ": "JWT", "kid": _jwk()["kid"]}
    payload = {"iss": client_id, "sub": client_id, "aud": EPIC_TOKEN_URL,
               "jti": str(uuid.uuid4()), "iat": now, "nbf": now, "exp": now + 240}
    signing_input = f"{_b64url(json.dumps(header).encode())}.{_b64url(json.dumps(payload).encode())}"
    sig = key.sign(signing_input.encode(), padding.PKCS1v15(), hashes.SHA384())
    return f"{signing_input}.{_b64url(sig)}"


def _epic_fetch_token(client_id: str) -> Dict:
    body = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
        "client_assertion": _epic_client_assertion(client_id),
    }).encode()
    req = urllib.request.Request(EPIC_TOKEN_URL, data=body,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:300]
        hint = ("Epic rejected the request. Common causes: (1) the JWK Set URL isn't publicly reachable yet, "
                "(2) Epic hasn't finished syncing the key (can take a few hours), "
                "(3) wrong client ID — use the NON-PRODUCTION one, "
                "(4) the app isn't a Backend Systems app.")
        raise HTTPException(502, f"Epic token request failed ({e.code}): {detail} — {hint}")
    except Exception as e:
        raise HTTPException(502, f"Could not reach Epic: {str(e)[:150]}")
    repo.set_setting("fhir_token", data["access_token"])
    repo.set_setting("fhir_base", EPIC_R4_BASE)
    repo.set_setting("epic_token_expires", str(int(time.time()) + int(data.get("expires_in", 3600))))
    return data


class EpicConnectRequest(BaseModel):
    client_id: str


@app.get("/v1/fhir/epic/status")
def epic_status(request: Request):
    exp = repo.get_setting("epic_token_expires", "")
    remaining = max(0, int(exp) - int(time.time())) if exp.isdigit() else 0
    base = str(request.base_url).rstrip("/")
    return {"client_id": repo.get_setting("epic_client_id", ""),
            "kid": _jwk()["kid"],
            "jwks_url": f"{base}/jwks.json",
            "jwks_public": not any(h in base for h in ("localhost", "127.0.0.1")),
            "token_valid_seconds": remaining,
            "connected": remaining > 0 and repo.get_setting("fhir_base", "") == EPIC_R4_BASE}


@app.post("/v1/fhir/epic/connect")
def epic_connect(body: EpicConnectRequest, x_user: Optional[str] = Header(None)):
    client_id = body.client_id.strip()
    if not client_id:
        raise HTTPException(422, "Client ID is required")
    repo.set_setting("epic_client_id", client_id)
    data = _epic_fetch_token(client_id)
    repo.audit(_actor(x_user), "fhir.epic_connected", "fhir", client_id,
               f"token valid {data.get('expires_in', '?')}s")
    return {"connected": True, "expires_in": data.get("expires_in"), "base_url": EPIC_R4_BASE,
            "scope": data.get("scope", "")}


@app.post("/v1/fhir/epic/refresh")
def epic_refresh(x_user: Optional[str] = Header(None)):
    client_id = repo.get_setting("epic_client_id", "")
    if not client_id:
        raise HTTPException(400, "Connect to Epic first (client ID not saved)")
    data = _epic_fetch_token(client_id)
    repo.audit(_actor(x_user), "fhir.epic_token_refreshed", "fhir", client_id, "")
    return {"connected": True, "expires_in": data.get("expires_in")}


def _fhir_get(path: str, _retry: bool = True) -> Dict:
    base = repo.get_setting("fhir_base", DEFAULT_FHIR_BASE).rstrip("/")
    token = repo.get_setting("fhir_token", "")
    headers = {"Accept": "application/fhir+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(base + path, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        # Epic tokens last ~1h — mint a fresh one automatically and retry once
        client_id = repo.get_setting("epic_client_id", "")
        if e.code in (401, 403) and client_id and _retry:
            try:
                _epic_fetch_token(client_id)
                return _fhir_get(path, _retry=False)
            except HTTPException:
                pass
        raise HTTPException(502, f"FHIR server returned {e.code} for {path} — check the base URL/token")
    except Exception as e:
        raise HTTPException(502, f"Could not reach FHIR server: {str(e)[:150]}")


def _fhir_patient_brief(p: Dict) -> Dict:
    name = ""
    if p.get("name"):
        n = p["name"][0]
        name = " ".join(n.get("given", []) + ([n.get("family")] if n.get("family") else [])) or n.get("text", "")
    return {"id": p.get("id"), "name": name or "(unnamed)", "gender": p.get("gender", "unknown"),
            "birthDate": p.get("birthDate", "")}


def _age_sex(p: Dict) -> str:
    age = ""
    if p.get("birthDate"):
        try:
            born = date.fromisoformat(p["birthDate"][:10])
            age = str((date.today() - born).days // 365)
        except ValueError:
            pass
    sex = (p.get("gender") or "u")[0].upper()
    return f"{age}{sex}" if age else sex


@app.get("/v1/fhir/config")
def fhir_config_get():
    return {"base_url": repo.get_setting("fhir_base", DEFAULT_FHIR_BASE),
            "has_token": bool(repo.get_setting("fhir_token", "")),
            "presets": FHIR_PRESETS}


@app.post("/v1/fhir/config")
def fhir_config_set(body: FhirConfigRequest, x_user: Optional[str] = Header(None)):
    if not body.base_url.startswith("https://"):
        raise HTTPException(422, "FHIR base URL must be https://")
    repo.set_setting("fhir_base", body.base_url.rstrip("/"))
    if body.token is not None:
        repo.set_setting("fhir_token", body.token.strip())
    repo.audit(_actor(x_user), "fhir.config_changed", "fhir", body.base_url, "token set" if body.token else "")
    return fhir_config_get()


@app.get("/v1/fhir/patients")
def fhir_patients(name: str = ""):
    q = f"/Patient?_count=10" + (f"&name={urllib.parse.quote(name)}" if name else "")
    bundle = _fhir_get(q)
    entries = bundle.get("entry", []) or []
    return [_fhir_patient_brief(e["resource"]) for e in entries if e.get("resource", {}).get("resourceType") == "Patient"]


@app.get("/v1/fhir/patients/{patient_id}/summary")
def fhir_patient_summary(patient_id: str):
    patient = _fhir_get(f"/Patient/{patient_id}")
    conditions, coverage = [], []
    try:
        cbundle = _fhir_get(f"/Condition?patient={urllib.parse.quote(patient_id)}&_count=10")
        for e in cbundle.get("entry", []) or []:
            r = e.get("resource", {})
            if r.get("resourceType") != "Condition":
                continue
            code = r.get("code", {})
            text = code.get("text") or (code.get("coding", [{}])[0].get("display", ""))
            icd = next((c.get("code") for c in code.get("coding", []) if "icd" in (c.get("system") or "").lower()), None)
            if text:
                conditions.append({"text": text, "icd10": icd,
                                   "status": (r.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", ""))})
    except HTTPException:
        pass
    try:
        vbundle = _fhir_get(f"/Coverage?patient={urllib.parse.quote(patient_id)}&_count=5")
        for e in vbundle.get("entry", []) or []:
            r = e.get("resource", {})
            if r.get("resourceType") != "Coverage":
                continue
            payor = (r.get("payor", [{}])[0].get("display")
                     or r.get("class", [{}])[0].get("name", "") if r.get("class") else "") or "Unknown payer"
            coverage.append({"payor": payor, "status": r.get("status", ""),
                             "plan": (r.get("class", [{}])[0].get("value", "") if r.get("class") else "")})
    except HTTPException:
        pass
    return {"patient": _fhir_patient_brief(patient), "age_sex": _age_sex(patient),
            "conditions": conditions, "coverage": coverage}


@app.post("/v1/fhir/import/{patient_id}")
def fhir_import(patient_id: str, x_user: Optional[str] = Header(None)):
    summary = fhir_patient_summary(patient_id)
    p = summary["patient"]
    cancer_cond = next((c for c in summary["conditions"]
                        if c.get("icd10", "") and c["icd10"][:1] in ("C", "D")), None) \
        or (summary["conditions"][0] if summary["conditions"] else None)
    cov = summary["coverage"][0] if summary["coverage"] else None
    payer_name = (cov or {}).get("payor", "")
    payer_id = next((pid for pid, n in PAYER_NAMES.items() if n.lower() in payer_name.lower()), "FHIR01")
    case_id = f"CASE-FHIR-{patient_id[:12]}"
    req = PrevisitCheckRequest(
        case_id=case_id, patient_ref=f"FHIR-{patient_id[:10]}", patient_name=p["name"],
        age_sex=summary["age_sex"],
        cancer=(cancer_cond or {}).get("text", "Condition not on file"),
        regimen="Imported from FHIR — schedule pending",
        physician="—", appt_time="09:00",
        payer_id=payer_id, plan_type="COMMERCIAL",
        appointment_date=date.today() + timedelta(days=3),
        cpt_codes_planned=[], referral_on_file=False, provider_npi="0000000000")
    resp = run_previsit_check(req)
    repo.audit(_actor(x_user), "fhir.patient_imported", "case", case_id, f"{p['name']} from FHIR")
    return {"case": resp, "source": {"patient_id": patient_id, "conditions": len(summary["conditions"]),
                                     "coverage": len(summary["coverage"])}}


# =========================================================================
# Seed — runs ONLY when the database is empty, so user edits persist.
# =========================================================================

@app.on_event("startup")
def seed():
    _seed_v2_additions()
    if repo.claims_count() > 1 or repo.previsit_count() > 0:
        return
    today = date.today()
    tomorrow = today + timedelta(days=1)
    in2days = today + timedelta(days=2)

    previsit = [
        PrevisitCheckRequest(case_id="CASE-2026-0201", patient_ref="PT-88213", patient_name="Amanda Stewart",
                             age_sex="47M", cancer="Brain (C71.9)", regimen="Chemo infusion — cycle 3",
                             physician="Dr. E. Rhodes · Med Onc", appt_time="08:30",
                             payer_id="TRI1", plan_type="GOVERNMENT", appointment_date=tomorrow,
                             cpt_codes_planned=["96413", "96415"], referral_on_file=True, provider_npi="1234567890"),
        PrevisitCheckRequest(case_id="CASE-2026-0202", patient_ref="PT-91027", patient_name="Maria Gonzalez",
                             age_sex="61F", cancer="Breast (C50.911)", regimen="Trastuzumab maintenance",
                             physician="Dr. A. Patel · Med Onc", appt_time="10:15",
                             payer_id="BCBS01", plan_type="COMMERCIAL", appointment_date=tomorrow,
                             cpt_codes_planned=["96413"], referral_on_file=True, provider_npi="1234567890"),
        PrevisitCheckRequest(case_id="CASE-2026-0203", patient_ref="PT-77012", patient_name="Robert King",
                             age_sex="66M", cancer="Colon (C18.9)", regimen="Bevacizumab infusion — 1st line",
                             physician="Dr. E. Rhodes · Med Onc", appt_time="13:00",
                             payer_id="UHC01", plan_type="COMMERCIAL", appointment_date=tomorrow,
                             cpt_codes_planned=["96413", "96415"], referral_on_file=False, provider_npi="1234567890"),
        PrevisitCheckRequest(case_id="CASE-2026-0204", patient_ref="PT-65544", patient_name="Linda Chen",
                             age_sex="54F", cancer="Ovarian (C56.9)", regimen="Carboplatin infusion",
                             physician="Dr. M. Osei · Med Onc", appt_time="09:00",
                             payer_id="AETNA1", plan_type="COMMERCIAL", appointment_date=in2days,
                             cpt_codes_planned=["96413", "96415"], referral_on_file=True, provider_npi="1234567890"),
        PrevisitCheckRequest(case_id="CASE-2026-0205", patient_ref="PT-70988", patient_name="James Turner",
                             age_sex="72M", cancer="Prostate (C61)", regimen="New-patient consult + labs",
                             physician="Dr. A. Patel · Med Onc", appt_time="11:30",
                             payer_id="HUM01", plan_type="MEDICARE_ADVANTAGE", appointment_date=in2days,
                             cpt_codes_planned=["99205"], referral_on_file=False, provider_npi="1234567890"),
        PrevisitCheckRequest(case_id="CASE-2026-0206", patient_ref="PT-83321", patient_name="David Okoro",
                             age_sex="59M", cancer="Lymphoma (C85.9)", regimen="Rituximab infusion",
                             physician="Dr. M. Osei · Med Onc", appt_time="15:45",
                             payer_id="MCO1", plan_type="MEDICAID", appointment_date=tomorrow,
                             cpt_codes_planned=["96413", "96415"], referral_on_file=False, provider_npi="1234567890"),
    ]
    for c in previsit:
        run_previsit_check(c)

    # Benefit accumulators (mock 271 benefit segments; null = unverifiable)
    repo.set_benefits("PT-88213", {"deductible_total": 1500, "deductible_used": 1500, "oop_max": 6000, "oop_used": 3420, "specialist_copay": 40, "coinsurance_pct": 10})
    repo.set_benefits("PT-91027", {"deductible_total": 2000, "deductible_used": 1150, "oop_max": 7500, "oop_used": 2210, "specialist_copay": 50, "coinsurance_pct": 20})
    repo.set_benefits("PT-77012", {"deductible_total": 3000, "deductible_used": 800, "oop_max": 8500, "oop_used": 800, "specialist_copay": 60, "coinsurance_pct": 20})
    repo.set_benefits("PT-65544", {"deductible_total": 1800, "deductible_used": 1800, "oop_max": 6500, "oop_used": 5100, "specialist_copay": 45, "coinsurance_pct": 15})
    repo.set_benefits("PT-70988", {"deductible_total": 500, "deductible_used": 500, "oop_max": 5500, "oop_used": 4890, "specialist_copay": 35, "coinsurance_pct": 20})
    # PT-83321 (David Okoro): coverage inactive — benefits unverifiable, none stored

    claims = [
        PrecheckRequest(claim_id="CLM001201", patient_name="Sarah M.", mrn="CLM001201", age_sex="58F",
                        stage="Stage IIIA", line="1st Line", plan_name="UHC Choice Plus PPO",
                        payer_id="UHC01", plan_type="COMMERCIAL", primary_dx="C34.10",
                        cpt_codes=["96413", "96415"], jcodes=["J9271"], billed_units=200,
                        billed_amount=28400.00, place_of_service="22", dos=date(2026, 6, 21), auth_number=None,
                        clinical_note="Stage IIIA lung carcinoma. PD-L1 TPS 65%. First-line pembrolizumab planned.",
                        provider_npi="1234567890"),
        PrecheckRequest(claim_id="CLM001188", patient_name="Robert K.", mrn="CLM001188", age_sex="66M",
                        stage="Stage IV", line="1st Line", plan_name="Humana Gold Plus",
                        payer_id="HUM01", plan_type="MEDICARE_ADVANTAGE", primary_dx="C18.9",
                        cpt_codes=["96413", "96415"], jcodes=["J9055"], billed_units=90,
                        billed_amount=22100.00, place_of_service="22", dos=date(2026, 6, 24), auth_number=None,
                        clinical_note="Metastatic colon adenocarcinoma. KRAS wild-type. Bevacizumab combination therapy.",
                        provider_npi="1234567890"),
        PrecheckRequest(claim_id="CLM001175", patient_name="James T.", mrn="CLM001175", age_sex="71M",
                        stage="Stage IV", line="2nd Line", plan_name="Aetna Open Choice",
                        payer_id="AETNA1", plan_type="COMMERCIAL", primary_dx="C61",
                        cpt_codes=["96413"], jcodes=["J9023"], billed_units=120,
                        billed_amount=14200.00, place_of_service="11", dos=date(2026, 6, 26),
                        auth_number="AUTH-88231",
                        clinical_note="Metastatic castration-resistant prostate cancer. PSA rising. Abiraterone + prednisone.",
                        provider_npi="1234567890"),
        PrecheckRequest(claim_id="CLM001163", patient_name="Linda C.", mrn="CLM001163", age_sex="54F",
                        stage="Stage IIIC", line="1st Line", plan_name="Meridian Advantage HMO",
                        payer_id="MERID1", plan_type="MEDICARE_ADVANTAGE", primary_dx="C56.9",
                        cpt_codes=["96413", "96415"], jcodes=["J9045"], billed_units=45,
                        billed_amount=12600.00, place_of_service="22", dos=date(2026, 6, 28),
                        auth_number="AUTH-90417",
                        clinical_note="Stage IIIC ovarian carcinoma. Carboplatin + paclitaxel. Biomarker panel pending.",
                        provider_npi="1234567890"),
        PrecheckRequest(claim_id="CLM001152", patient_name="Maria G.", mrn="CLM001152", age_sex="61F",
                        stage="Stage II", line="Maintenance", plan_name="BCBS PPO",
                        payer_id="BCBS01", plan_type="COMMERCIAL", primary_dx="C50.911",
                        cpt_codes=["96413"], jcodes=["J9355"], billed_units=40,
                        billed_amount=9800.00, place_of_service="11", dos=date(2026, 6, 30),
                        auth_number="AUTH-77120",
                        clinical_note="HER2 IHC 3+ confirmed on pathology. Trastuzumab maintenance, cycle 8 of 17.",
                        provider_npi="1234567890"),
    ]
    for c in claims:
        run_precheck(c)

    _post_seed_story()


def _seed_v2_additions():
    """Idempotent additions that apply to existing databases too (no wipe needed)."""
    # Demo login account (username: admin / password: precheck123)
    repo.create_user("admin", "precheck123", role="admin")
    if repo.get_claim("CLM001147") is None:
        run_precheck(PrecheckRequest(
            claim_id="CLM001147", patient_name="Elena P.", mrn="CLM001147", age_sex="63F",
            stage="Stage III", line="2nd Line", plan_name="BCBS PPO",
            payer_id="BCBS01", plan_type="COMMERCIAL", primary_dx="R51.9",
            cpt_codes=["96415", "9641X"], jcodes=["J9312"],
            modifiers=[["JW", "JW", None, None]], billed_units=70,
            billed_amount=6800.00, place_of_service="22", dos=date(2026, 7, 2),
            auth_number="AUTH-40118",
            clinical_note="Follicular lymphoma, maintenance rituximab. Coding review pending.",
            provider_npi="1234567890"))


def _post_seed_story():
    # Story seed exercising the full closed loop:
    # 1. Sarah M.'s claim was denied by the payer (CARC 197) → appears in Appeals.
    repo.deny_claim("CLM001201", "197", "N702")
    repo.audit("seed", "claim.denied", "claim", "CLM001201", "CARC 197")
    for exc in repo.list_exceptions(case_id="CLM001201"):
        if exc.category == ExceptionCategory.AUTH:
            repo.add_evidence(exc.exception_id, EvidenceCreate(
                payer_source="UHC Provider Portal", verification_method=VerificationMethod.PORTAL,
                reference_number="UHC-AUTH-55821", verified_by="A. Patel"), actor="seed")
    # 2. Robert K.'s auth blocker was caught and resolved BEFORE submission → a prevented denial.
    for exc in repo.list_exceptions(case_id="CLM001188"):
        if exc.category == ExceptionCategory.AUTH:
            repo.add_evidence(exc.exception_id, EvidenceCreate(
                payer_source="Humana Auth Line", verification_method=VerificationMethod.PHONE,
                reference_number="HUM-770-441", verified_by="R. Juarez"), actor="seed")
            repo.update_exception(exc.exception_id, ExceptionUpdate(
                owner="R. Juarez", status=ExceptionStatus.RESOLVED,
                resolution_notes="Auth obtained by phone before submission; approval linked to claim."), actor="seed")
