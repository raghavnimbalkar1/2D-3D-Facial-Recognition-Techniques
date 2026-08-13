"""Split correctness: zero leakage, reproducibility, pair counts."""

from __future__ import annotations

from ivafr.datasets.splits import assert_no_leakage, make_split


def _ids(split, key):
    return set(split[key])


def test_p1_no_overlap(toy_manifest):
    for seed in range(3):
        s = make_split(toy_manifest, "P1_closed", seed=seed)
        assert not (_ids(s, "gallery_ids") & _ids(s, "probe_ids"))
        assert len(s["gallery_ids"]) > 0 and len(s["probe_ids"]) > 0


def test_p1_gallery_is_frontal_normal(toy_manifest):
    s = make_split(toy_manifest, "P1_closed", seed=0)
    gm = toy_manifest[toy_manifest["sample_id"].astype(str).isin(s["gallery_ids"])]
    assert (gm["pose_yaw"] == 0).all()
    assert (gm["pose_pitch"] == 0).all()
    assert (gm["illumination"] == "normal").all()


def test_p2_subject_disjoint(toy_manifest):
    for seed in range(3):
        s = make_split(toy_manifest, "P2_disjoint", seed=seed)
        assert not (set(s["train_subjects"]) & set(s["eval_subjects"]))
        assert len(s["train_subjects"]) >= 1 and len(s["eval_subjects"]) >= 1
        assert_no_leakage(s)


def test_p2_probe_gallery_only_eval_subjects(toy_manifest):
    s = make_split(toy_manifest, "P2_disjoint", seed=0)
    eval_subj = set(s["eval_subjects"])
    gm = toy_manifest[toy_manifest["sample_id"].astype(str).isin(s["gallery_ids"] + s["probe_ids"])]
    assert set(gm["subject_id"]) == eval_subj


def test_verification_no_self_pairs(toy_manifest):
    s = make_split(toy_manifest, "P1_closed", seed=0)
    for a, b in s["verification"]["genuine"] + s["verification"]["impostor"]:
        assert a != b


def test_genuine_pairs_share_subject(toy_manifest):
    subj = dict(zip(toy_manifest["sample_id"].astype(str), toy_manifest["subject_id"]))
    s = make_split(toy_manifest, "P1_closed", seed=0)
    for a, b in s["verification"]["genuine"]:
        assert subj[a] == subj[b]


def test_impostor_ratio(toy_manifest):
    s = make_split(toy_manifest, "P2_disjoint", seed=0)
    g, i = len(s["verification"]["genuine"]), len(s["verification"]["impostor"])
    assert i <= 20 * g + 1


def test_reproducible_under_seed(toy_manifest):
    a = make_split(toy_manifest, "P2_disjoint", seed=3)
    b = make_split(toy_manifest, "P2_disjoint", seed=3)
    assert a["train_subjects"] == b["train_subjects"]
    assert a["gallery_ids"] == b["gallery_ids"]
    assert a["verification"]["genuine"] == b["verification"]["genuine"]
    assert a["verification"]["impostor"] == b["verification"]["impostor"]


def test_different_seeds_differ(toy_manifest):
    a = make_split(toy_manifest, "P2_disjoint", seed=0)
    b = make_split(toy_manifest, "P2_disjoint", seed=1)
    assert a["train_subjects"] != b["train_subjects"]