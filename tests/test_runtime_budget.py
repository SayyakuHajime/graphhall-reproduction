"""Resource-budget regression: the encoder must share smoke's CPU limit."""

from graphhall import cli


def test_smoke_sets_thread_budget_before_encoding(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        cli.torch, "set_num_threads", lambda value: events.append(("threads", value))
    )
    monkeypatch.setattr(cli, "prepare_command", lambda args: 0)
    monkeypatch.setattr(cli, "encode_command", lambda args: events.append(("encode", None)))
    monkeypatch.setattr(cli, "run_command", lambda args: 0)

    assert cli.main(["smoke", "--threads", "3", "--output-dir", str(tmp_path)]) == 0
    assert events == [("threads", 3), ("encode", None)]
