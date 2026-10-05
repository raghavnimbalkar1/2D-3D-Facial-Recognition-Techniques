import pytest
from ivafr.datasets.eligibility import assert_eligible, eligible_samples


def test_single_mesh_subjects_cannot_be_evaluated(toy_manifest):
    frame = toy_manifest.groupby("subject_id").head(1)
    with pytest.raises(ValueError, match="Independent"):
        assert_eligible(frame)


def test_quality_rejections_do_not_enter_experiments(toy_manifest):
    with pytest.raises(ValueError, match="No accepted"):
        eligible_samples(toy_manifest, "2d")
    frame = toy_manifest.copy()
    frame["detect_ok"] = True
    frame["align_ok"] = True
    frame.loc[0, "detect_ok"] = False
    accepted = eligible_samples(frame, "2d")
    assert frame.iloc[0].sample_id not in set(accepted.sample_id)
