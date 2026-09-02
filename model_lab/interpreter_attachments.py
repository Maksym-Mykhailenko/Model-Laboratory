"""Deterministic, local-only reference attachments for interpreter authoring.

The language model never receives paths or raw file bytes.  Supported files are decoded or
text-extracted in the persistent scientific sidecar, content-addressed, split into deterministic
UTF-8 chunks, and exposed to inference only through the bounded interpreter context protocol.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import importlib.metadata
import io
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .canonical import canonical_json_sha256


ATTACHMENT_SCHEMA = "model-laboratory-interpreter-attachment"
ATTACHMENT_SCHEMA_VERSION = "1.0"
ATTACHMENT_CHUNKING_ALGORITHM = "utf8-codepoint-chunks-v1"
MAX_ACTIVE_ATTACHMENTS = 8
MAX_ATTACHMENT_SOURCE_BYTES = 16 * 1024 * 1024
MAX_ATTACHMENT_TEXT_BYTES = 8 * 1024 * 1024
MAX_ATTACHMENT_STORE_TEXT_BYTES = 32 * 1024 * 1024
MAX_ATTACHMENT_PDF_PAGES = 2_000
ATTACHMENT_CHUNK_BYTES = 3 * 1024

_MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".json": "application/json",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".pdf": "application/pdf",
}


class InterpreterAttachmentError(ValueError):
    """An attachment is unsupported, malformed, excessive, or no longer available."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalise_text(value: str) -> str:
    if "\x00" in value:
        raise InterpreterAttachmentError("Reference attachments must not contain NUL characters.")
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _validated_name(value: object) -> tuple[str, str, str]:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise InterpreterAttachmentError("Attachment filename must be a non-empty string.")
    if (
        value in {".", ".."}
        or any(character in "/\\" or ord(character) < 32 for character in value)
        or Path(value).name != value
        or len(value.encode("utf-8")) > 255
    ):
        raise InterpreterAttachmentError(
            "Attachment filename must contain only one portable filename component."
        )
    suffix = Path(value).suffix.casefold()
    media_type = _MEDIA_TYPES.get(suffix)
    if media_type is None:
        raise InterpreterAttachmentError(
            "Unsupported attachment type. Use TXT, Markdown, YAML, JSON, CSV, TSV, or a text-based PDF."
        )
    return value, suffix, media_type


def _decode_base64(value: object) -> bytes:
    if not isinstance(value, str) or not value:
        raise InterpreterAttachmentError("Attachment bytes must be supplied as base64.")
    estimated = (len(value) // 4) * 3
    if estimated > MAX_ATTACHMENT_SOURCE_BYTES + 2:
        raise InterpreterAttachmentError("Attachment exceeds the 16 MiB source-file limit.")
    try:
        decoded = base64.b64decode(value.encode("ascii"), validate=True)
    except (ValueError, UnicodeEncodeError, binascii.Error) as exc:
        raise InterpreterAttachmentError("Attachment payload is not valid base64.") from exc
    if not decoded:
        raise InterpreterAttachmentError("Empty files cannot be attached for interpretation.")
    if len(decoded) > MAX_ATTACHMENT_SOURCE_BYTES:
        raise InterpreterAttachmentError("Attachment exceeds the 16 MiB source-file limit.")
    return decoded


def _utf8_chunks(text: str) -> list[str]:
    chunks: list[str] = []
    characters: list[str] = []
    size = 0
    for character in text:
        encoded_size = len(character.encode("utf-8"))
        if characters and size + encoded_size > ATTACHMENT_CHUNK_BYTES:
            chunks.append("".join(characters))
            characters = []
            size = 0
        characters.append(character)
        size += encoded_size
    if characters or not chunks:
        chunks.append("".join(characters))
    return chunks


def _extract_pdf_text(
    source: bytes,
    *,
    reader_factory: Callable[[io.BytesIO], object] | None = None,
    version: str | None = None,
) -> tuple[str, dict[str, Any]]:
    if not source.startswith(b"%PDF-"):
        raise InterpreterAttachmentError("The selected .pdf file has no PDF signature.")
    if reader_factory is None:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise InterpreterAttachmentError(
                "PDF text extraction is unavailable because the pypdf dependency is not installed."
            ) from exc

        reader_factory = lambda stream: PdfReader(stream, strict=False)
        try:
            version = importlib.metadata.version("pypdf")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    try:
        reader = reader_factory(io.BytesIO(source))
        encrypted = bool(getattr(reader, "is_encrypted"))
    except Exception as exc:
        raise InterpreterAttachmentError(f"The PDF could not be parsed safely: {exc}") from exc
    if encrypted:
        raise InterpreterAttachmentError("Encrypted PDFs cannot be attached.")
    try:
        pages = list(getattr(reader, "pages"))
    except Exception as exc:
        raise InterpreterAttachmentError(f"The PDF pages could not be read safely: {exc}") from exc
    if not pages:
        raise InterpreterAttachmentError("The PDF contains no pages.")
    if len(pages) > MAX_ATTACHMENT_PDF_PAGES:
        raise InterpreterAttachmentError(
            f"The PDF exceeds the {MAX_ATTACHMENT_PDF_PAGES}-page attachment limit."
        )
    extracted_pages: list[str] = []
    extracted_bytes = 0
    for index, page in enumerate(pages):
        try:
            value = page.extract_text()
        except Exception as exc:
            raise InterpreterAttachmentError(
                f"Text extraction failed on PDF page {index + 1}: {exc}"
            ) from exc
        page_text = _normalise_text(value or "")
        extracted_pages.append(page_text)
        extracted_bytes += len(page_text.encode("utf-8"))
        if index:
            extracted_bytes += 3  # UTF-8 bytes in the deterministic \n\f\n separator.
        if extracted_bytes > MAX_ATTACHMENT_TEXT_BYTES:
            raise InterpreterAttachmentError("Extracted PDF text exceeds the 8 MiB text limit.")
    text = "\n\f\n".join(extracted_pages)
    if not text.strip():
        raise InterpreterAttachmentError(
            "No selectable text was found. Scanned/image-only PDFs require a future OCR or vision layer."
        )
    return text, {
        "method": "pypdf-page-text-v1",
        "implementation": "pypdf",
        "implementation_version": str(version or "unknown"),
        "page_count": len(pages),
    }


def _manifest_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": ATTACHMENT_SCHEMA,
        "schema_version": ATTACHMENT_SCHEMA_VERSION,
        "name": record["name"],
        "media_type": record["media_type"],
        "source_size_bytes": record["source_size_bytes"],
        "source_sha256": record["source_sha256"],
        "extracted_text_bytes": record["extracted_text_bytes"],
        "extracted_text_sha256": record["extracted_text_sha256"],
        "extraction": dict(record["extraction"]),
        "chunking": dict(record["chunking"]),
    }


def attachment_manifest(record: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_attachment_record(record)
    return {
        "attachment_id": validated["attachment_id"],
        **_manifest_payload(validated),
    }


def validate_attachment_manifest(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InterpreterAttachmentError("Attachment manifest must be an object.")
    required = {
        "attachment_id",
        "schema",
        "schema_version",
        "name",
        "media_type",
        "source_size_bytes",
        "source_sha256",
        "extracted_text_bytes",
        "extracted_text_sha256",
        "extraction",
        "chunking",
    }
    if set(value) != required:
        raise InterpreterAttachmentError("Attachment manifest fields are invalid.")
    name, _suffix, media_type = _validated_name(value.get("name"))
    if value.get("schema") != ATTACHMENT_SCHEMA or value.get(
        "schema_version"
    ) != ATTACHMENT_SCHEMA_VERSION:
        raise InterpreterAttachmentError("Unsupported attachment manifest schema.")
    if value.get("media_type") != media_type:
        raise InterpreterAttachmentError("Attachment media type does not match its filename.")
    for field, maximum in (
        ("source_size_bytes", MAX_ATTACHMENT_SOURCE_BYTES),
        ("extracted_text_bytes", MAX_ATTACHMENT_TEXT_BYTES),
    ):
        candidate = value.get(field)
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, int)
            or not 0 < candidate <= maximum
        ):
            raise InterpreterAttachmentError(f"Attachment {field} is invalid.")
    for field in ("source_sha256", "extracted_text_sha256", "attachment_id"):
        candidate = value.get(field)
        if (
            not isinstance(candidate, str)
            or len(candidate) != 64
            or any(character not in "0123456789abcdef" for character in candidate)
        ):
            raise InterpreterAttachmentError(f"Attachment {field} is not a SHA-256 digest.")
    extraction = value.get("extraction")
    if not isinstance(extraction, Mapping) or set(extraction) != {
        "method",
        "implementation",
        "implementation_version",
        "page_count",
    }:
        raise InterpreterAttachmentError("Attachment extraction identity is invalid.")
    if any(
        not isinstance(extraction.get(field), str) or not extraction[field]
        for field in ("method", "implementation", "implementation_version")
    ):
        raise InterpreterAttachmentError("Attachment extraction identity is incomplete.")
    page_count = extraction.get("page_count")
    if page_count is not None and (
        isinstance(page_count, bool)
        or not isinstance(page_count, int)
        or not 1 <= page_count <= MAX_ATTACHMENT_PDF_PAGES
    ):
        raise InterpreterAttachmentError("Attachment PDF page count is invalid.")
    chunking = value.get("chunking")
    if not isinstance(chunking, Mapping) or set(chunking) != {
        "algorithm",
        "maximum_chunk_bytes",
        "chunk_count",
    }:
        raise InterpreterAttachmentError("Attachment chunking identity is invalid.")
    if (
        chunking.get("algorithm") != ATTACHMENT_CHUNKING_ALGORITHM
        or chunking.get("maximum_chunk_bytes") != ATTACHMENT_CHUNK_BYTES
        or isinstance(chunking.get("chunk_count"), bool)
        or not isinstance(chunking.get("chunk_count"), int)
        or chunking["chunk_count"] < 1
    ):
        raise InterpreterAttachmentError("Attachment chunking contract is invalid.")
    result = {
        "attachment_id": value["attachment_id"],
        **_manifest_payload(
            {
                **value,
                "name": name,
                "media_type": media_type,
                "extraction": dict(extraction),
                "chunking": dict(chunking),
            }
        ),
    }
    expected_id = canonical_json_sha256(_manifest_payload(result))
    if result["attachment_id"] != expected_id:
        raise InterpreterAttachmentError("Attachment identifier does not match its manifest.")
    return result


def validate_attachment_manifests(values: object) -> list[dict[str, Any]]:
    if not isinstance(values, list) or len(values) > MAX_ACTIVE_ATTACHMENTS:
        raise InterpreterAttachmentError(
            f"Attachment manifest must contain at most {MAX_ACTIVE_ATTACHMENTS} files."
        )
    result = sorted(
        (validate_attachment_manifest(item) for item in values),
        key=lambda item: item["attachment_id"],
    )
    identifiers = [item["attachment_id"] for item in result]
    if len(identifiers) != len(set(identifiers)):
        raise InterpreterAttachmentError("Attachment manifest contains duplicate files.")
    return result


def validate_attachment_record(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "attachment_id",
        "schema",
        "schema_version",
        "name",
        "media_type",
        "source_size_bytes",
        "source_sha256",
        "extracted_text",
        "extracted_text_bytes",
        "extracted_text_sha256",
        "extraction",
        "chunking",
    }:
        raise InterpreterAttachmentError("Attachment record fields are invalid.")
    text = value.get("extracted_text")
    if not isinstance(text, str) or not text:
        raise InterpreterAttachmentError("Attachment record contains no extracted text.")
    text = _normalise_text(text)
    encoded = text.encode("utf-8")
    manifest = validate_attachment_manifest(
        {key: item for key, item in value.items() if key != "extracted_text"}
    )
    chunks = _utf8_chunks(text)
    if (
        len(encoded) != manifest["extracted_text_bytes"]
        or _sha256_bytes(encoded) != manifest["extracted_text_sha256"]
        or len(chunks) != manifest["chunking"]["chunk_count"]
    ):
        raise InterpreterAttachmentError("Attachment extracted-text identity is inconsistent.")
    return {**manifest, "extracted_text": text}


def attachment_chunks(record: Mapping[str, Any]) -> list[str]:
    return _utf8_chunks(validate_attachment_record(record)["extracted_text"])


def ingest_attachment(name: object, data_base64: object) -> dict[str, Any]:
    filename, suffix, media_type = _validated_name(name)
    source = _decode_base64(data_base64)
    if suffix == ".pdf":
        text, extraction = _extract_pdf_text(source)
    else:
        try:
            text = _normalise_text(source.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise InterpreterAttachmentError(
                "Text and structured attachments must be valid UTF-8."
            ) from exc
        if not text:
            raise InterpreterAttachmentError("The attachment contains no text.")
        extraction = {
            "method": "utf8-decode-newline-normalisation-v1",
            "implementation": "model-laboratory",
            "implementation_version": "1.0",
            "page_count": None,
        }
    encoded_text = text.encode("utf-8")
    if len(encoded_text) > MAX_ATTACHMENT_TEXT_BYTES:
        raise InterpreterAttachmentError("Extracted attachment text exceeds the 8 MiB limit.")
    chunks = _utf8_chunks(text)
    partial: dict[str, Any] = {
        "schema": ATTACHMENT_SCHEMA,
        "schema_version": ATTACHMENT_SCHEMA_VERSION,
        "name": filename,
        "media_type": media_type,
        "source_size_bytes": len(source),
        "source_sha256": _sha256_bytes(source),
        "extracted_text_bytes": len(encoded_text),
        "extracted_text_sha256": _sha256_bytes(encoded_text),
        "extraction": extraction,
        "chunking": {
            "algorithm": ATTACHMENT_CHUNKING_ALGORITHM,
            "maximum_chunk_bytes": ATTACHMENT_CHUNK_BYTES,
            "chunk_count": len(chunks),
        },
    }
    record = {
        "attachment_id": canonical_json_sha256(partial),
        **partial,
        "extracted_text": text,
    }
    return validate_attachment_record(record)


class InterpreterAttachmentStore:
    """Bounded in-memory store owned by the persistent scientific sidecar."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}

    def add(self, record: Mapping[str, Any]) -> dict[str, Any]:
        validated = validate_attachment_record(record)
        identifier = validated["attachment_id"]
        projected_total = sum(
            item["extracted_text_bytes"] for key, item in self._records.items() if key != identifier
        ) + validated["extracted_text_bytes"]
        if projected_total > MAX_ATTACHMENT_STORE_TEXT_BYTES:
            raise InterpreterAttachmentError(
                "Active attachment text exceeds the 32 MiB sidecar memory limit. Remove a file first."
            )
        self._records[identifier] = validated
        return attachment_manifest(validated)

    def remove(self, attachment_id: object) -> bool:
        if not isinstance(attachment_id, str):
            raise InterpreterAttachmentError("attachment_id must be a string.")
        return self._records.pop(attachment_id, None) is not None

    def resolve(self, attachment_ids: object) -> tuple[dict[str, Any], ...]:
        if not isinstance(attachment_ids, list) or len(attachment_ids) > MAX_ACTIVE_ATTACHMENTS:
            raise InterpreterAttachmentError(
                f"At most {MAX_ACTIVE_ATTACHMENTS} attachments may be active."
            )
        if any(not isinstance(item, str) for item in attachment_ids):
            raise InterpreterAttachmentError("Every attachment identifier must be a string.")
        if len(attachment_ids) != len(set(attachment_ids)):
            raise InterpreterAttachmentError("Active attachment identifiers must be unique.")
        missing = [item for item in attachment_ids if item not in self._records]
        if missing:
            raise InterpreterAttachmentError(
                "An attached reference is no longer available in the scientific sidecar. Reattach it."
            )
        return tuple(self._records[item] for item in sorted(attachment_ids))

    def clear(self) -> None:
        self._records.clear()
