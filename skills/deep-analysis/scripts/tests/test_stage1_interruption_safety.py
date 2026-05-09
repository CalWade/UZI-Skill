"""Regression test for Bug 2 · stage1 interruption safety (v3.3.4).

Scenario addressed
------------------
`collect_raw_data` in `run_real_test.py` flushes `raw_data.json` every 3 fetchers.
If the process is killed mid-wave (say after 6 / 20 fetchers complete), the next
read sees a raw_data.json whose `dimensions` dict has 6 keys — which was
indistinguishable from "fewer-than-expected but legitimate cache" before this
patch.

Downstream consequences observed in the wild:
- Agents read the truncated cache and believed stage1 was complete.
- `self-review` flagged `missing dim N_` only because the dim-count check.
- Users hit `UZI_SKIP_REVIEW=1` to bypass → HTML generated with only half the
  data.

Fix
---
1. `collect_raw_data` marks `raw["_stage1_in_progress"] = True` at entry and
   clears it after wave 3 completes successfully.
2. `self_review.check_stage1_not_interrupted` reports a critical issue if the
   flag is still set — forcing the user to `--no-resume`.
3. The resume loader in `collect_raw_data` treats `_stage1_in_progress=True`
   cache as invalid and re-fetches instead of stitching onto half-done state.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_check_exists_and_registered():
    """New check must be importable AND registered in CHECKS."""
    from lib import self_review
    assert hasattr(self_review, "check_stage1_not_interrupted"), (
        "check_stage1_not_interrupted must be defined in lib/self_review.py"
    )
    assert self_review.check_stage1_not_interrupted in self_review.CHECKS, (
        "check_stage1_not_interrupted must be registered in CHECKS list"
    )


def test_check_raises_critical_when_flag_set():
    """raw has _stage1_in_progress=True → one critical issue."""
    from lib.self_review import check_stage1_not_interrupted

    ctx = {
        "raw": {
            "_stage1_in_progress": True,
            "_stage1_expected_dims": [
                "0_basic", "1_financials", "2_kline", "3_macro",
                "4_peers", "5_chain", "6_research",
            ],
        },
        "dims": {"0_basic": {"data": {}}, "1_financials": {"data": {}}},
    }
    issues = check_stage1_not_interrupted(ctx)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.severity == "critical"
    assert issue.dim == "stage1"
    assert "_stage1_in_progress" in issue.issue or "未完整收尾" in issue.issue
    # 缺失信息 evidence 中要体现 missing count
    assert "missing" in issue.evidence.lower() or "预期" in issue.evidence


def test_check_silent_when_flag_absent():
    """Normal completed raw_data (no flag) → zero issues."""
    from lib.self_review import check_stage1_not_interrupted

    ctx = {
        "raw": {"ticker": "002545.SZ", "market": "A"},
        "dims": {"0_basic": {"data": {"name": "东方铁塔"}}},
    }
    issues = check_stage1_not_interrupted(ctx)
    assert issues == [], f"expected no issues, got {issues}"


def test_check_silent_when_flag_false():
    """Explicit flag=False still means stage1 completed."""
    from lib.self_review import check_stage1_not_interrupted

    ctx = {
        "raw": {"_stage1_in_progress": False},
        "dims": {},
    }
    issues = check_stage1_not_interrupted(ctx)
    assert issues == []


def test_review_all_bubbles_stage1_flag():
    """End-to-end: a cache with _stage1_in_progress=True must fail review."""
    from lib.self_review import review_all
    from unittest.mock import patch

    def fake_read(ticker, name):
        if name == "raw_data":
            return {
                "ticker": ticker, "market": "A",
                "_stage1_in_progress": True,
                "_stage1_expected_dims": ["0_basic", "1_financials"],
                "dimensions": {"0_basic": {"data": {"name": "东方铁塔"}}},
            }
        return {}

    with patch("lib.cache.read_task_output", side_effect=fake_read):
        report = review_all("002545.SZ")

    assert report["critical_count"] >= 1
    stage1_issues = [i for i in report["issues"] if i.get("dim") == "stage1"]
    assert len(stage1_issues) == 1, (
        f"expected exactly one stage1 issue, got {stage1_issues}"
    )


def test_collect_raw_data_has_flag_in_source():
    """run_real_test.collect_raw_data sets+clears the flag (source-level check).

    We don't run stage1 live (needs network + 3 min). Just assert the code
    contains the flag set and the flag clear.
    """
    rrt_path = SCRIPTS_DIR / "run_real_test.py"
    source = rrt_path.read_text(encoding="utf-8")
    assert 'raw["_stage1_in_progress"] = True' in source, (
        "collect_raw_data must set _stage1_in_progress=True at entry"
    )
    assert 'raw.pop("_stage1_in_progress"' in source, (
        "collect_raw_data must pop _stage1_in_progress at successful end"
    )
    assert "_stage1_expected_dims" in source


def test_resume_loader_discards_in_progress_cache():
    """Resume logic must refuse to reuse a cache with _stage1_in_progress=True."""
    rrt_path = SCRIPTS_DIR / "run_real_test.py"
    source = rrt_path.read_text(encoding="utf-8")
    # a sentinel string matching our patch — this proves the safety branch exists
    assert 'prev.get("_stage1_in_progress") is True' in source, (
        "Resume logic must detect interrupted prior run and bypass cache"
    )
