from __future__ import annotations

import base64
from copy import deepcopy

import pytest

from model_lab.interpreter_attachments import (
    ATTACHMENT_CHUNK_BYTES,
    InterpreterAttachmentError,
    InterpreterAttachmentStore,
    _extract_pdf_text,
    attachment_chunks,
    attachment_manifest,
    ingest_attachment,
    validate_attachment_record,
)


def _encoded(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def test_utf8_reference_is_normalised_content_addressed_and_chunked() -> None:
    text = ("alpha\r\nbeta\r" + "x" * (ATTACHMENT_CHUNK_BYTES + 40)).encode()
    record = ingest_attachment("notes.md", _encoded(text))
    manifest = attachment_manifest(record)

    assert record["extracted_text"].startswith("alpha\nbeta\n")
    assert manifest["name"] == "notes.md"
    assert manifest["media_type"] == "text/markdown"
    assert manifest["attachment_id"] == record["attachment_id"]
    assert len(attachment_chunks(record)) == 2
    assert all(len(chunk.encode("utf-8")) <= ATTACHMENT_CHUNK_BYTES for chunk in attachment_chunks(record))


def test_attachment_records_and_store_fail_closed_on_tampering_or_missing_ids() -> None:
    record = ingest_attachment("model.yaml", _encoded(b"name: reference\n"))
    altered = deepcopy(record)
    altered["extracted_text"] = "name: forged\n"
    with pytest.raises(InterpreterAttachmentError, match="identity is inconsistent"):
        validate_attachment_record(altered)

    store = InterpreterAttachmentStore()
    manifest = store.add(record)
    assert store.resolve([manifest["attachment_id"]])[0]["name"] == "model.yaml"
    with pytest.raises(InterpreterAttachmentError, match="no longer available"):
        store.resolve(["0" * 64])
    assert store.remove(manifest["attachment_id"]) is True


def test_unsupported_binary_and_invalid_utf8_are_rejected() -> None:
    with pytest.raises(InterpreterAttachmentError, match="Unsupported attachment type"):
        ingest_attachment("image.png", _encoded(b"not an image"))
    with pytest.raises(InterpreterAttachmentError, match="valid UTF-8"):
        ingest_attachment("notes.txt", _encoded(b"\xff\xfe"))


class _Page:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class _Reader:
    def __init__(self, pages: list[_Page], *, encrypted: bool = False) -> None:
        self.pages = pages
        self.is_encrypted = encrypted


def test_pdf_text_extraction_is_page_ordered_and_identified() -> None:
    text, identity = _extract_pdf_text(
        b"%PDF-1.7 fake",
        reader_factory=lambda _stream: _Reader([_Page("first\r\n"), _Page("second")]),
        version="test-version",
    )
    assert text == "first\n\n\f\nsecond"
    assert identity == {
        "method": "pypdf-page-text-v1",
        "implementation": "pypdf",
        "implementation_version": "test-version",
        "page_count": 2,
    }


def test_scanned_or_encrypted_pdf_is_rejected_without_ocr() -> None:
    with pytest.raises(InterpreterAttachmentError, match="No selectable text"):
        _extract_pdf_text(
            b"%PDF-1.7 fake",
            reader_factory=lambda _stream: _Reader([_Page(None)]),
        )
    with pytest.raises(InterpreterAttachmentError, match="Encrypted PDFs"):
        _extract_pdf_text(
            b"%PDF-1.7 fake",
            reader_factory=lambda _stream: _Reader([_Page("hidden")], encrypted=True),
        )
