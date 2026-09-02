"""YAML loading for Model Laboratory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .schema import ModelSpec


class ModelParseError(ValueError):
    """Raised when a model file cannot be read as a valid external specification."""


def parse_model_data(data: Any) -> ModelSpec:
    """Validate already-loaded YAML data against the external model schema."""
    if not isinstance(data, dict):
        raise ModelParseError("The model document must contain a YAML mapping at its top level.")

    try:
        return ModelSpec.model_validate(data)
    except ValidationError as exc:
        raise ModelParseError(str(exc)) from exc


def parse_model_text(text: str) -> ModelSpec:
    """Parse YAML text and validate it against the external model schema."""
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ModelParseError(f"Invalid YAML: {exc}") from exc
    return parse_model_data(data)


def load_model_file(path: str | Path) -> ModelSpec:
    """Load and validate a YAML model file from disk."""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ModelParseError(f"Could not read model file '{file_path}': {exc}") from exc
    return parse_model_text(text)
