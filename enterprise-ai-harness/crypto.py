"""
Wrapper around the external data masking service.

Comments and docstrings translated to English for review; logic unchanged.

Endpoints:
    POST /api/v1/tokenize  — forward direction (names → tokens).
    POST /api/v1/restore   — reverse direction (tokens → names).

Files are passed by their path on disk (storageRef), so the service needs
access to STORAGE_DIR. Files are not cleaned up yet — they accumulate.
"""

import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

# Configuration
TOKENIZE_URL = os.getenv("CRYPTO_TOKENIZE_URL",
                         "http://masking-service.example/api/v1/tokenize")
RESTORE_URL  = os.getenv("CRYPTO_RESTORE_URL",
                         "http://masking-service.example/api/v1/restore")
AUDIT_URL    = os.getenv("CRYPTO_AUDIT_URL",
                         "http://masking-service.example/api/v1/audit")
STORAGE_DIR  = Path(os.getenv("CRYPTO_STORAGE_DIR",
                              "/data/masking-service/files"))
TIMEOUT      = float(os.getenv("CRYPTO_TIMEOUT_SEC", "60"))


class CryptoError(RuntimeError):
    """Failure while calling the masking service."""


# Extension -> (attachment type, content_type).
# TABLES (CSV/XLSX) are accepted by the masking service as a file, and it returns a masked copy.
# DOCUMENTS (PDF/DOCX/TXT) are NOT accepted as an attachment (verified on dev: HTTP 400),
# so no masked copy of them exists — their text is extracted locally and masked at the
# tool-result barrier (llm.tokenize_text, "tool result"). See read_uploaded.
FILE_TYPES: dict[str, tuple[str, str]] = {
    ".csv": ("CSV", "text/csv"),
    ".xlsx": ("XLSX",
              "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ".xls": ("XLSX",
             "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
    ".pdf": ("PDF", "application/pdf"),
    ".docx": ("DOCX",
              "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    ".txt": ("TXT", "text/plain"),
}

# Types the masking service does not accept as an attachment (see the comment above).
DOCUMENT_FILE_TYPES = frozenset({"PDF", "DOCX", "TXT"})


@dataclass
class Attachment:
    """An attachment as described in a request/response."""
    attachment_id: str
    file_name: str
    storage_ref: str
    content_type: str = "text/csv"
    file_type: str = "CSV"             # CSV | XLSX | PDF | DOCX | TXT
    status: Optional[str] = None       # MASKED | RESTORED (response only)


def is_document(attachment: Any) -> bool:
    """A document (PDF/DOCX/TXT) — it must not be sent to tokenize as an attachment."""
    return (getattr(attachment, "file_type", "") or "").upper() in DOCUMENT_FILE_TYPES


@dataclass
class TokenizeResult:
    request_id: str
    session_id: str
    blocked: bool
    block_reasons: list[str]
    masked_prompt: str
    masked_attachments: list[Attachment]
    detected_entities: list[dict]
    processing_time_ms: int
    raw: dict = field(default_factory=dict)  # full service response, for debugging


@dataclass
class RestoreResult:
    request_id: str
    session_id: str
    restored_text: str
    restored_attachments: list[Attachment]
    processing_time_ms: int
    raw: dict = field(default_factory=dict)


# =============================================================================
# Low-level POST with error handling
# =============================================================================

def _post(url: str, payload: dict) -> dict:
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise CryptoError(f"network error: {e}") from e
    if r.status_code != 200:
        raise CryptoError(f"{url} {r.status_code}: {r.text[:500]}")
    try:
        return r.json()
    except ValueError as e:
        raise CryptoError(f"bad json: {e}") from e


# =============================================================================
# Tokenize
# =============================================================================

def tokenize(
    prompt: str,
    user_id: str,
    request_id: Optional[str] = None,
    purpose: str = "SALES_FORECAST",
    department: str = "SALES",
    external_llm: bool = True,
    llm_model: Optional[str] = None,
    attachments: Optional[list[Attachment]] = None,
) -> TokenizeResult:
    """
    Forward tokenization of text (and, optionally, of attachments).
    Returns a TokenizeResult carrying everything the service reported.
    If blocked=True the result must not be sent to an LLM; show the reason to the user.
    """
    rid = request_id or str(uuid.uuid4())

    payload = {
        "requestId": rid,
        "userId": user_id,
        "purpose": purpose,
        "prompt": prompt or "",
        "attachments": [
            {
                "attachmentId": a.attachment_id,
                "fileName": a.file_name,
                "contentType": a.content_type,
                "type": a.file_type,
                "storageRef": a.storage_ref,
            }
            for a in (attachments or [])
        ],
        "context": {
            "source": "LOCAL_FILE" if attachments else "INLINE",
            "department": department,
            "externalLlm": "true" if external_llm else "false",
            **({"llmModel": llm_model} if llm_model else {}),
        },
    }

    data = _post(TOKENIZE_URL, payload)

    return TokenizeResult(
        request_id=data.get("requestId", rid),
        session_id=data.get("sessionId", ""),
        blocked=bool(data.get("blocked", False)),
        block_reasons=list(data.get("blockReasons", []) or []),
        masked_prompt=data.get("maskedPrompt", ""),
        masked_attachments=[
            Attachment(
                attachment_id=a["attachmentId"],
                file_name=a.get("maskedFileName", a.get("originalFileName", "")),
                storage_ref=a.get("storageRef", ""),
                content_type=a.get("contentType", "text/csv"),
                file_type=a.get("type", "CSV"),
                status=a.get("status"),
            )
            for a in data.get("maskedAttachments", []) or []
        ],
        detected_entities=list(data.get("detectedEntities", []) or []),
        processing_time_ms=int(data.get("processingTimeMs", 0)),
        raw=data,
    )


# =============================================================================
# Restore
# =============================================================================

def restore(
    text: str,
    session_id: str,
    request_id: str,
    attachments: Optional[list[Attachment]] = None,
) -> RestoreResult:
    """
    Reverse tokenization. session_id and request_id come from the preceding tokenize.
    """
    payload = {
        "requestId": request_id,
        "sessionId": session_id,
        "text": text or "",
        "attachments": [
            {
                "attachmentId": a.attachment_id,
                "fileName": a.file_name,
                "contentType": a.content_type,
                "type": a.file_type,
                "storageRef": a.storage_ref,
            }
            for a in (attachments or [])
        ],
    }

    data = _post(RESTORE_URL, payload)

    return RestoreResult(
        request_id=data.get("requestId", request_id),
        session_id=data.get("sessionId", session_id),
        restored_text=data.get("restoredText", ""),
        restored_attachments=[
            Attachment(
                attachment_id=a["attachmentId"],
                file_name=a.get("maskedFileName", ""),
                storage_ref=a.get("storageRef", ""),
                content_type=a.get("contentType", "text/csv"),
                file_type=a.get("type", "CSV"),
                status=a.get("status"),
            )
            for a in data.get("restoredAttachments", []) or []
        ],
        processing_time_ms=int(data.get("processingTimeMs", 0)),
        raw=data,
    )


# =============================================================================
# File helpers
# =============================================================================

def save_uploaded_file(uploaded_file, prefix: str = "upload") -> Attachment:
    """
    Save a streamlit UploadedFile into STORAGE_DIR and return an Attachment.
    The name is made unique with a uuid so that uploads cannot collide.
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    ext = Path(uploaded_file.name).suffix.lower()
    unique_name = f"{prefix}-{uuid.uuid4().hex[:8]}{ext}"
    dst = STORAGE_DIR / unique_name
    with open(dst, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)

    file_type, content_type = FILE_TYPES.get(ext, ("CSV", "text/csv"))

    return Attachment(
        attachment_id=f"att-{uuid.uuid4().hex[:8]}",
        file_name=uploaded_file.name,
        storage_ref=str(dst),
        content_type=content_type,
        file_type=file_type,
    )


def is_configured() -> bool:
    """Check that the endpoints are configured."""
    return bool(TOKENIZE_URL and RESTORE_URL)


def health_check() -> tuple[bool, str]:
    """
    Simple availability probe: performs a short tokenize.
    Returns (ok, message).
    """
    try:
        res = tokenize(
            prompt="ping",
            user_id="healthcheck",
            request_id=f"hc-{uuid.uuid4().hex[:8]}",
        )
        return True, f"OK ({res.processing_time_ms} ms, session={res.session_id[:8]}...)"
    except CryptoError as e:
        return False, str(e)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

# =============================================================================
# Audit
# =============================================================================

def list_audit_events(
    page: int = 0,
    size: int = 50,
    status: Optional[str] = None,           # SUCCESS | BLOCKED | FAILED
    operation: Optional[str] = None,        # TOKENIZE | RESTORE
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    created_from: Optional[str] = None,     # ISO 8601, e.g. 2026-05-06T00:00:00+03:00
    created_to: Optional[str] = None,
) -> dict:
    """
    Audit events with filtering and pagination.
    Returns the service dict as is: items, page, size, totalElements, totalPages, hasNext.
    """
    params = {"page": page, "size": size}
    if status:        params["status"] = status
    if operation:     params["operation"] = operation
    if request_id:    params["requestId"] = request_id
    if session_id:    params["sessionId"] = session_id
    if user_id:       params["userId"] = user_id
    if created_from:  params["createdFrom"] = created_from
    if created_to:    params["createdTo"] = created_to

    try:
        r = requests.get(f"{AUDIT_URL}/events", params=params, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise CryptoError(f"audit network error: {e}") from e
    if r.status_code != 200:
        raise CryptoError(f"audit {r.status_code}: {r.text[:500]}")
    return r.json()


def get_audit_event(event_id: str, include_archive_ref: bool = False) -> dict:
    """A single audit event."""
    params = {"includeArchiveRef": "true"} if include_archive_ref else None
    try:
        r = requests.get(f"{AUDIT_URL}/events/{event_id}",
                         params=params, timeout=TIMEOUT)
    except requests.RequestException as e:
        raise CryptoError(f"audit network error: {e}") from e
    if r.status_code == 404:
        raise CryptoError(f"audit event not found: {event_id}")
    if r.status_code != 200:
        raise CryptoError(f"audit {r.status_code}: {r.text[:500]}")
    return r.json()


def get_archive_url(archive_token: str) -> str:
    """Download URL for the event ZIP archive (hand it to the user via st.link_button)."""
    return f"{AUDIT_URL}/archives/{archive_token}"