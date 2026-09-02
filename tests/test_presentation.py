from __future__ import annotations

from model_lab.presentation import format_number


def test_format_number_collapses_numerical_zero_noise() -> None:
    assert format_number(-1e-14) == "0"
    assert format_number(7e-12) == "0"


def test_format_number_keeps_clean_values_compact() -> None:
    assert format_number(1.0) == "1"
    assert format_number(-4.71238898038469) == "-4.712389"
