"""Stage 1/2 — Preprocessing: 2D chain + 3D chain, content-hash cached."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from ivafr.datasets.manifest import read_manifest, write_manifest
from ivafr.logging_utils import get_logger
from ivafr.preprocess import cache as pcache
from ivafr.preprocess.align2d import TEMPLATE_112, align_to_template, to_gray
from ivafr.preprocess.detect import detect_face
from ivafr.preprocess.illum import normalize_illum
from ivafr.preprocess.range_image import range_image_from_depth
from ivafr.preprocess.normals import normals_from_depth
from ivafr.preprocess.curvature import curvature_from_depth
from ivafr.registry import get_dataset

log = get_logger("pipelines.preprocess")


def _out_2d(interim: Path, subject: str, sample: str, suffix: str) -> Path:
    d = interim / "crops2d" / subject
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sample}{suffix}"


def _out_3d(interim: Path, subject: str, sample: str, suffix: str) -> Path:
    d = interim / "range" / subject
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sample}{suffix}"


def _out_lmk(interim: Path, subject: str, sample: str) -> Path:
    d = interim / "landmarks3d" / subject
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sample}_lmk3d.npy"


def preprocess_2d_sample(
    img: np.ndarray,
    gt_landmarks: np.ndarray | None,
    cfg2d: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """2D chain: detect -> align -> illumnorm -> grayscale/normalise.

    Returns:
        (img112 uint8 3ch, gray64 float32, detect_ok, detection source).
    """
    det = detect_face(img, gt_landmarks, cfg2d.get("detector", {}))
    if not det.ok:
        return np.zeros((112, 112, 3), np.uint8), np.zeros((64, 64), np.float32), False, "none"
    alg = cfg2d.get("align", {})
    img112 = align_to_template(img, det.landmarks, size=112, template=np.asarray(alg.get("template", TEMPLATE_112), np.float32))
    gray64 = to_gray(cv2.resize(img112, (64, 64), interpolation=cv2.INTER_AREA))

    method = cfg2d.get("illum", {}).get("method", "none")
    params = cfg2d.get("illum", {}).get("params", {})
    gray64 = normalize_illum(gray64, method, **params)
    return img112, gray64, True, det.source


def preprocess_3d_sample(depth: np.ndarray, cfg3d: dict[str, Any]) -> tuple[np.ndarray, bool, float]:
    """3D chain (toy): depth map -> range image + hole ratio."""
    size = int(cfg3d.get("range", {}).get("size", 64))
    fill = cfg3d.get("range", {}).get("fill", "nearest")
    z_norm = cfg3d.get("range", {}).get("z_norm", "std")
    rimg, hole = range_image_from_depth(depth, size=size, fill=fill, z_norm=z_norm)
    if not np.isfinite(rimg).all():
        log.warning("Range image contains non-finite values")
        return rimg, False, hole
    return rimg, True, hole


def _pseudo_depth_from_facemesh(image: np.ndarray, cfg3d: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build a sparse image-space depth field from FaceMesh landmarks.

    FaceMesh provides fixed semantic correspondence. The sparse field is
    filled only after projection; no sensor point-cloud parser is involved.
    """
    from ivafr.preprocess.landmarks3d import facemesh_landmarks, nose_tip

    lcfg = cfg3d.get("landmarks", {})
    pts = facemesh_landmarks(
        image,
        refine=bool(lcfg.get("refine_landmarks", True)),
        model_path=lcfg.get("model_asset_path"),
    )
    h, w = image.shape[:2]
    depth = np.full((h, w), np.nan, dtype=np.float32)
    xx = np.clip(np.rint(pts[:, 0] * (w - 1)).astype(int), 0, w - 1)
    yy = np.clip(np.rint(pts[:, 1] * (h - 1)).astype(int), 0, h - 1)
    depth[yy, xx] = pts[:, 2]
    return depth, pts


def preprocess_dataset(
    dataset: str,
    data_root: str | Path,
    cfg2d: dict[str, Any],
    cfg3d: dict[str, Any],
    modality: str = "both",
    limit: int | None = None,
) -> pd.DataFrame:
    """Run both preprocessing chains over a manifest; returns updated manifest.

    Content-hash caching makes re-runs near-instant.
    """
    data_root = Path(data_root)
    manifest_path = data_root / "processed" / dataset / "manifest.csv"
    manifest = read_manifest(manifest_path)
    adapter_cls = get_dataset(dataset)
    adapter = adapter_cls(raw_root=data_root / "raw")
    samples_by_id = {s.sample_id: s for s in adapter.discover()}
    interim = data_root / "interim" / dataset
    interim.mkdir(parents=True, exist_ok=True)

    rows = manifest if limit is None else manifest.head(limit)
    for i, row in rows.iterrows():
        subject, sample_id = str(row["subject_id"]), str(row["sample_id"])
        sample = samples_by_id.get(sample_id)
        if sample is None:
            log.error("Sample %s missing from adapter output", sample_id)
            continue

        if modality in ("2d", "both") and row["has_2d"]:
            img = adapter.load_2d(sample)
            gt_lms = adapter.load_landmarks(sample)
            cfg_key = {"2d": cfg2d}
            in_digest = pcache.file_digest(sample.path_2d)
            out_png = _out_2d(interim, subject, sample_id, "_a112.png")
            out_npy = _out_2d(interim, subject, sample_id, "_g64.npy")
            if not pcache.is_cached(out_png, cfg_key, in_digest):
                img112, gray64, ok, src = preprocess_2d_sample(img, gt_lms, cfg2d)
                cv2.imwrite(str(out_png), img112)
                np.save(out_npy, gray64)
                pcache.mark_cached(out_png, cfg_key, in_digest)
                pcache.mark_cached(out_npy, cfg_key, in_digest)
                manifest.at[i, "detect_ok"] = ok
                manifest.at[i, "align_ok"] = ok
                log.debug("%s 2d detect ok=%s src=%s", sample_id, ok, src)

        if modality in ("3d", "both") and row["has_3d"]:
            pseudo = sample.path_3d is None
            if pseudo:
                depth, lmk3d = _pseudo_depth_from_facemesh(adapter.load_2d(sample), cfg3d)
            else:
                depth = np.load(sample.path_3d).astype(np.float32)
            cfg_key = {"3d": cfg3d}
            in_digest = pcache.file_digest(sample.path_2d if pseudo else sample.path_3d)
            out_npy = _out_3d(interim, subject, sample_id, "_r64.npy")
            if not pcache.is_cached(out_npy, cfg_key, in_digest):
                rimg, ok, hole = preprocess_3d_sample(depth, cfg3d)
                np.save(out_npy, rimg)
                size = rimg.shape[0]
                normals = normals_from_depth(rimg)
                curv = curvature_from_depth(rimg)
                out_norm = _out_3d(interim, subject, sample_id, "_n64.npy")
                out_curv = _out_3d(interim, subject, sample_id, "_c64.npy")
                np.save(out_norm, normals)
                np.save(out_curv, curv)
                lms = adapter.load_landmarks(sample)
                if pseudo:
                    np.save(_out_lmk(interim, subject, sample_id), lmk3d.astype(np.float32))
                elif lms is not None and np.asarray(lms).ndim == 2 and np.asarray(lms).shape[1] == 2:
                    ll = np.asarray(lms, dtype=np.float32).copy()
                    h, w = depth.shape
                    yy = np.clip(np.rint(ll[:, 1]).astype(int), 0, h - 1)
                    xx = np.clip(np.rint(ll[:, 0]).astype(int), 0, w - 1)
                    lmk3d = np.column_stack([(ll[:, 0] - w / 2) / max(w, 1),
                                              (ll[:, 1] - h / 2) / max(h, 1),
                                              np.nan_to_num(depth[yy, xx], nan=0.0)])
                    np.save(_out_lmk(interim, subject, sample_id), lmk3d.astype(np.float32))
                pcache.mark_cached(out_npy, cfg_key, in_digest)
                manifest.at[i, "nosetip_ok"] = ok
                log.debug("%s 3d ok=%s hole=%.3f", sample_id, ok, hole)

    if modality == "2d":
        manifest["quality_flag"] = np.where(manifest["detect_ok"], "", "rejected")
    elif modality == "3d":
        manifest["quality_flag"] = np.where(manifest["nosetip_ok"], "", "rejected")
    else:
        manifest["quality_flag"] = np.where(
            manifest["detect_ok"] & manifest["nosetip_ok"], "", "rejected"
        )
    reject = int((manifest["quality_flag"] == "rejected").sum())
    write_manifest(manifest, manifest_path)
    log.info(
        "Preprocessing done: rejected=%d / %d rows",
        reject,
        len(manifest),
    )
    return manifest
