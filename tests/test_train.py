import pytest

pytest.importorskip("torch")
pytest.importorskip("tensorboard")

from ghostrl.train import Config, train  # noqa: E402


def test_stop_file_ends_a_run_early_and_logs_are_written(tmp_path):
    cfg = Config(maze="small", num_envs=8, num_steps=16, total_steps=10_000, eval_games=4, out=str(tmp_path))
    seen = []

    def on_update(u, out):
        seen.append(u)
        if u == 2:
            (out / "STOP").write_text("")

    result = train(cfg, on_update=on_update)
    assert seen[:3] == [1, 2, -1]  # stopped after update 2, then the final callback
    assert (tmp_path / "checkpoint.pt").exists() and (tmp_path / "metrics.csv").exists()
    assert list((tmp_path / "tb").glob("events.*"))
    assert "greedy" in result
