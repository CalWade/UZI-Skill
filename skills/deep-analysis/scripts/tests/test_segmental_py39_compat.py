"""Regression test for lib/report/segmental.py · Py3.9 f-string compat.

Bug (pre-fix): line 236 had nested f-string with backslash escapes:
    f'  {f"<div class=\"seg-metrics-row\">{margin_badges}</div>" if margin_badges else ""}'

Python 3.9 raises `SyntaxError: f-string expression part cannot include a backslash`
so `assemble_report` silently skipped the whole segmental block with a warning.
Fix: extract the conditional into a pre-computed local variable.

This test guards against reintroduction via two checks:
1. Module imports cleanly on Py3.9 (no SyntaxError).
2. `_render_segmental_block` can be called without raising.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_segmental_module_imports_on_py39():
    """Regression: segmental.py must not have f-string backslash escapes."""
    # Force fresh import to catch SyntaxError that would have surfaced at module load.
    mod_name = "lib.report.segmental"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    mod = importlib.import_module(mod_name)
    assert hasattr(mod, "_render_segmental_block"), (
        "_render_segmental_block export missing after import"
    )


def test_render_segmental_block_handles_missing_model(tmp_path, monkeypatch):
    """`_render_segmental_block` must return empty string when no segmental_model.json exists."""
    # Point cwd at an empty tmp dir so .cache/{ticker}/segmental_model.json is absent.
    monkeypatch.chdir(tmp_path)
    from lib.report.segmental import _render_segmental_block

    out = _render_segmental_block("NONEXISTENT.SZ")
    assert out == "", f"expected empty string for missing model, got {len(out)} chars"


def test_render_segmental_block_does_not_raise_syntax_error():
    """Calling the function must not raise SyntaxError regardless of input."""
    from lib.report.segmental import _render_segmental_block

    # Even with a nonsense ticker it should not SyntaxError — at worst returns "".
    try:
        _render_segmental_block("")
    except SyntaxError:
        raise AssertionError("segmental block raised SyntaxError — Py3.9 bug reintroduced")
    except Exception:
        # Other exceptions are out of scope for this regression test.
        pass
