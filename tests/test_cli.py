from typer.testing import CliRunner

from ivafr.cli import app


def test_environment_default_and_cli_precedence(monkeypatch, tmp_path):
    received = []
    monkeypatch.setenv("IVAFR_DATA_DIR", str(tmp_path / "env"))
    monkeypatch.setattr(
        "ivafr.datasets.toy.generate_toy", lambda root, **kwargs: received.append(root)
    )
    runner = CliRunner()
    assert runner.invoke(app, ["dataset-build"]).exit_code == 0
    assert received[-1] == tmp_path / "env/raw"
    assert (
        runner.invoke(app, ["dataset-build", "--data-root", str(tmp_path / "flag")]).exit_code == 0
    )
    assert received[-1] == tmp_path / "flag/raw"


def test_cannot_generate_a_real_dataset():
    result = CliRunner().invoke(app, ["dataset-build", "--name", "tufts3d"])
    assert result.exit_code != 0
    assert "synthetic toy" in result.output


def test_robustness_cannot_report_success_without_runs(tmp_path):
    result = CliRunner().invoke(
        app, ["robustness", "--exp", "E14", "--results-root", str(tmp_path)]
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
