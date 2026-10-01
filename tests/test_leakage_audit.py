import pytest
from ivafr.datasets.splits import make_split, assert_no_leakage


@pytest.mark.parametrize(
    "corruption", ["train_id", "genuine_label", "impostor_label", "unknown_id", "stale"]
)
def test_corrupt_split_fails(toy_manifest, corruption):
    split = make_split(toy_manifest, "P2_disjoint", 0)
    if corruption == "train_id":
        split["train_ids"].append(split["probe_ids"][0])
    if corruption == "genuine_label":
        split["verification"]["genuine"][0] = split["verification"]["impostor"][0]
    if corruption == "impostor_label":
        split["verification"]["impostor"][0] = split["verification"]["genuine"][0]
    if corruption == "unknown_id":
        split["probe_ids"].append("unknown")
    if corruption == "stale":
        split["manifest_hash"] = "old"
    with pytest.raises(AssertionError):
        assert_no_leakage(split, toy_manifest)


def test_shared_capture_rejected(toy_manifest):
    frame = toy_manifest.copy()
    split = make_split(frame, "P1_closed", 0)
    ids = [split["gallery_ids"][0], split["probe_ids"][0]]
    frame.loc[frame.sample_id.isin(ids), "capture_id"] = "same_acquisition"
    with pytest.raises(AssertionError, match="Shared source"):
        make_split(frame, "P1_closed", 0)
