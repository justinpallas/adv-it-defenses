"""Unit tests for naming helpers."""

from __future__ import annotations

from advdef.utils.naming import build_identifier, slugify_label


def test_slugify_label_basic():
    assert slugify_label("JPEG Defense 75%") == "jpeg-defense-75"


def test_slugify_label_default_fallback():
    assert slugify_label("   !!!   ", default="fallback") == "fallback"


def test_build_identifier_prefers_name():
    ident = build_identifier(name="Linfty Attack", params={"eps": 0.05}, default_prefix="autoattack")
    assert ident == "linfty-attack"


def test_build_identifier_hashes_params_when_name_missing():
    ident_a = build_identifier(name=None, params={"eps": 0.05}, default_prefix="autoattack")
    ident_b = build_identifier(name=None, params={"eps": 3.0}, default_prefix="autoattack")

    assert ident_a.startswith("autoattack-")
    assert ident_b.startswith("autoattack-")
    assert ident_a != ident_b
