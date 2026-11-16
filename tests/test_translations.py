"""Tests that every translation key used by the integration is translated."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

COMPONENT_ROOT = Path(__file__).parent.parent
COMPONENT_PATH = COMPONENT_ROOT / "custom_components" / "local_openai"
TRANSLATIONS_PATH = COMPONENT_PATH / "translations" / "en.json"
STRINGS_PATH = COMPONENT_PATH / "strings.json"

# Check every source of strings that exists. strings.json is optional for
# custom integrations, but when it is present it has to stay in sync.
TRANSLATION_FILES = [
    path for path in (TRANSLATIONS_PATH, STRINGS_PATH) if path.exists()
]


def _translation_key_usages() -> list[tuple[str, int, str]]:
    """Return the file, line and key of every literal translation_key usage."""
    usages: list[tuple[str, int, str]] = []

    for path in sorted(COMPONENT_PATH.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for keyword in node.keywords:
                if (
                    keyword.arg == "translation_key"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ):
                    usages.append(
                        (
                            str(path.relative_to(COMPONENT_PATH)),
                            node.lineno,
                            keyword.value.value,
                        )
                    )

    return usages


def _load_translations(path: Path) -> dict[str, Any]:
    """Load one translation file."""
    return json.loads(path.read_text(encoding="utf-8"))


def test_translation_keys_are_scanned() -> None:
    """The scanner finds the known exception key used by the entity."""
    keys = {key for _, _, key in _translation_key_usages()}
    assert "unsupported_attachment_type" in keys


def test_english_translations_exist() -> None:
    """The runtime translation file is present and valid JSON."""
    assert TRANSLATIONS_PATH.is_file()
    assert isinstance(_load_translations(TRANSLATIONS_PATH), dict)


@pytest.mark.parametrize(
    "translation_file",
    TRANSLATION_FILES,
    ids=[path.name for path in TRANSLATION_FILES],
)
def test_every_translation_key_is_translated(translation_file: Path) -> None:
    """Every literal translation_key usage has an exceptions message."""
    translations = _load_translations(translation_file)
    exceptions = translations.get("exceptions", {})

    missing = [
        f"{source}:{line} uses translation_key '{key}'"
        for source, line, key in _translation_key_usages()
        if (
            not isinstance(exceptions.get(key), dict)
            or "message" not in exceptions[key]
        )
    ]

    relative_path = translation_file.relative_to(COMPONENT_PATH)
    message = f"Missing exceptions entries in {relative_path}: "
    assert not missing, message + "; ".join(missing)
