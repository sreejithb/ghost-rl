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


def test_resume_from_checkpoint_loads_weights_and_continues_the_step_count(tmp_path):
    import torch

    first = Config(maze="small", num_envs=8, num_steps=16, total_steps=256, eval_games=4, out=str(tmp_path / "a"))
    train(first)
    ck = torch.load(tmp_path / "a" / "checkpoint.pt", map_location="cpu")
    assert ck["steps"] == 256
    second = Config(maze="small", num_envs=8, num_steps=16, total_steps=256, eval_games=4, out=str(tmp_path / "b"),
                    init_from=str(tmp_path / "a" / "checkpoint.pt"))
    train(second)
    ck2 = torch.load(tmp_path / "b" / "checkpoint.pt", map_location="cpu")
    assert ck2["steps"] == 512
    last = (tmp_path / "b" / "metrics.csv").read_text().strip().splitlines()[-1].split(",")
    assert int(last[1]) == 512
