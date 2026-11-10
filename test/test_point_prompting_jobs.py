import importlib
from pathlib import Path

import pytest


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("evaluate_point_prompting")


def test_cooperative_scoring_excludes_duplicate_writers(runner, tmp_path):
    result = tmp_path / "video.json"
    with runner.claim_video(result) as first:
        assert first
        with runner.claim_video(result) as second:
            assert not second
        runner.save(result, {"score": 1})
    with runner.claim_video(result) as completed:
        assert not completed


def test_failed_video_remains_claimable_and_existing_plan_is_checked(runner, tmp_path):
    result = tmp_path / "video.json"
    with pytest.raises(RuntimeError, match="interrupted"), runner.claim_video(result) as claimed:
        assert claimed
        raise RuntimeError("interrupted")
    with runner.claim_video(result) as retry:
        assert retry

    plan = tmp_path / "plan.json"
    runner.ensure_plan(plan, {"frames": 12})
    runner.ensure_plan(plan, {"frames": 12})
    with pytest.raises(ValueError, match="Mismatched experiment"):
        runner.ensure_plan(plan, {"frames": 9})
