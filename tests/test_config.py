from pathlib import Path
import shutil
import pytest
import yaml
from ivafr.config import ConfigResolver

CONFIGS = Path(__file__).parents[1] / "configs"


def test_yaml_defaults_and_inline_override(tmp_path):
    shutil.copytree(CONFIGS, tmp_path / "configs")
    root = tmp_path / "configs"
    p = root / "features/pca.yaml"
    settings = yaml.safe_load(p.read_text())
    settings["params"]["max_components"] = 17
    p.write_text(yaml.safe_dump(settings))
    resolver = ConfigResolver(root)
    assert resolver.experiment("E11").arms[0].feature_params["max_components"] == 17
    raw = yaml.safe_load((root / "experiments/E11.yaml").read_text())
    raw["arms"][0]["feature_params"] = {"max_components": 7}
    assert resolver._resolve(raw).arms[0].feature_params["max_components"] == 7


@pytest.mark.parametrize(
    "mutation", ["bad_param", "duplicate_seed", "duplicate_arm", "bad_protocol", "bad_count"]
)
def test_invalid_config_rejected(mutation):
    raw = yaml.safe_load((CONFIGS / "experiments/E11.yaml").read_text())
    if mutation == "bad_param":
        raw["arms"][0]["feature_params"] = {"typo": 3}
    if mutation == "duplicate_seed":
        raw["seeds"] = [0, 0]
    if mutation == "duplicate_arm":
        raw["arms"].append(raw["arms"][0])
    if mutation == "bad_protocol":
        raw["protocols"] = ["unknown"]
    if mutation == "bad_count":
        raw["arms"][0]["feature_params"] = {"max_components": 0}
    with pytest.raises(ValueError):
        ConfigResolver(CONFIGS)._resolve(raw)


def test_fusion_placeholder_fails_before_data_access():
    with pytest.raises(ValueError, match="Fusion"):
        ConfigResolver(CONFIGS).experiment("E04")
