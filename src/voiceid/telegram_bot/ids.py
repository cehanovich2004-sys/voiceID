"""Pseudonymous identifier generation for local Telegram collection."""

from __future__ import annotations

import secrets
from typing import Final

_HEX_BYTES: Final = 16


def new_sample_id() -> str:
    """Return a Phase 5B-compatible pseudonymous sample identifier."""

    return f"smp_{secrets.token_hex(_HEX_BYTES)}"


def new_subject_id() -> str:
    """Return a Phase 5B-compatible pseudonymous subject identifier."""

    return f"sub_{secrets.token_hex(_HEX_BYTES)}"


def new_source_group_id() -> str:
    """Return a Phase 5B-compatible pseudonymous source-group identifier."""

    return f"src_{secrets.token_hex(_HEX_BYTES)}"
