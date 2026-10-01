"""Stage 1/2 — Preprocessing: 2D chain + 3D chain, content-hash cached."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import io

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
from ivafr.integrity import atomic_bytes, digest, file_hash

log = get_logger("pipelines.preprocess")
PREPROCESS_VERSION = 2


def preprocessing_key(modality: str, config: dict) -> dict:
    """Invalidate cached representations when their producing code changes."""
    root = Path(__file__).parents[1]
    sources = [Path(__file__), *sorted((root / "preprocess").glob("*.py"))]
    return {
        modality: config,
        "version": PREPROCESS_VERSION,
        "code": digest({p.relative_to(root).as_posix(): file_hash(p) for p in sources}),
    }


def save_array(path: Path, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, array)
    atomic_bytes(path, buffer.getvalue())


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
    img112 = align_to_template(
        img,
        det.landmarks,
        size=112,
        template=np.asarray(alg.get("template", TEMPLATE_112), np.float32),
    )
    gray64 = to_gray(cv2.resize(img112, (64, 64), interpolation=cv2.INTER_AREA))

    method = cfg2d.get("illum", {}).get("method", "none")
    params = cfg2d.get("illum", {}).get("params", {})
    gray64 = normalize_illum(gray64, method, **params)
    return img112, gray64, True, det.source


def preprocess_3d_sample(
    depth: np.ndarray, cfg3d: dict[str, Any]
) -> tuple[np.ndarray, bool, float]:
    """3D chain (toy): depth map -> range image + hole ratio."""
    size = int(cfg3d.get("range", {}).get("size", 64))
    fill = cfg3d.get("range", {}).get("fill", "nearest")
    z_norm = cfg3d.get("range", {}).get("z_norm", "std")
    rimg, hole = range_image_from_depth(depth, size=size, fill=fill, z_norm=z_norm)
    max_holes = float(cfg3d.get("quality", {}).get("max_hole_ratio", 0.98))
    if not np.isfinite(rimg).all() or hole > max_holes or np.std(rimg) < 1e-6:
        log.warning("Range image contains non-finite values")
        return rimg, False, hole
    return rimg, True, hole


def _pseudo_depth_from_facemesh(
    image: np.ndarray, cfg3d: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Build a sparse image-space depth field from FaceMesh landmarks.

    FaceMesh provides fixed semantic correspondence. The sparse field is
    filled only after projection; no sensor point-cloud parser is involved.
    """
    from ivafr.preprocess.landmarks3d import facemesh_landmarks

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
    if modality not in {"2d", "3d", "both"}:
        raise ValueError("modality must be 2d, 3d, or both")
    manifest_path = data_root / "processed" / dataset / "manifest.csv"
    manifest = read_manifest(manifest_path)
    for column in (
        "detect_source",
        "reject_reason_2d",
        "reject_reason_3d",
        "preprocess_2d_hash",
        "preprocess_3d_hash",
    ):
        if column not in manifest:
            manifest[column] = ""
    adapter_cls = get_dataset(dataset)
    adapter = adapter_cls(raw_root=data_root / "raw")
    samples_by_id = {s.sample_id: s for s in adapter.discover()}
    interim = data_root / "interim" / dataset
    interim.mkdir(parents=True, exist_ok=True)

    rows = manifest if limit is None else manifest.head(limit)
    key2d = preprocessing_key("2d", cfg2d)
    key3d = preprocessing_key("3d", cfg3d)
    for i, row in rows.iterrows():
        subject, sample_id = str(row["subject_id"]), str(row["sample_id"])
        sample = samples_by_id.get(sample_id)
        if sample is None:
            log.error("Sample %s missing from adapter output", sample_id)
            continue

        if modality in ("2d", "both") and row["has_2d"]:
            img = adapter.load_2d(sample)
            gt_lms = adapter.load_landmarks(sample)
            cfg_key = key2d
            source_hash = pcache.file_digest(sample.path_2d)
            in_digest = digest(
                [
                    source_hash,
                    pcache.file_digest(sample.path_landmarks) if sample.path_landmarks else "",
                ]
            )
            out_png = _out_2d(interim, subject, sample_id, "_a112.png")
            out_npy = _out_2d(interim, subject, sample_id, "_g64.npy")
            if row["preprocess_2d_hash"] != pcache.cfg_hash(cfg_key) or not all(
                pcache.is_cached(p, cfg_key, in_digest) for p in (out_png, out_npy)
            ):
                img112, gray64, ok, src = preprocess_2d_sample(img, gt_lms, cfg2d)
                ok = bool(ok and np.isfinite(gray64).all() and np.std(gray64) > 1e-6)
                encoded_ok, encoded = cv2.imencode(".png", img112)
                if not encoded_ok:
                    raise IOError(f"Cannot encode crop {sample_id}")
                atomic_bytes(out_png, encoded.tobytes())
                save_array(out_npy, gray64)
                pcache.mark_cached(out_png, cfg_key, in_digest)
                pcache.mark_cached(out_npy, cfg_key, in_digest)
                manifest.at[i, "detect_ok"] = ok
                manifest.at[i, "align_ok"] = ok
                manifest.at[i, "detect_source"] = src
                manifest.at[i, "reject_reason_2d"] = "" if ok else "detection_or_degenerate_crop"
                log.debug("%s 2d detect ok=%s src=%s", sample_id, ok, src)
            manifest.at[i, "preprocess_2d_hash"] = pcache.cfg_hash(cfg_key)
            manifest.at[i, "content_hash_2d"] = source_hash

        if modality in ("3d", "both") and row["has_3d"]:
            pseudo = sample.path_3d is None
            if pseudo:
                depth, lmk3d = _pseudo_depth_from_facemesh(adapter.load_2d(sample), cfg3d)
            elif sample.path_3d and str(sample.path_3d).lower().endswith(".ply"):
                cloud = adapter.load_3d(sample)
                from ivafr.preprocess.mesh_to_depth import mesh_to_depth_map

                grid_sz = int(cfg3d.get("range", {}).get("size", 64))
                mesh_cfg = cfg3d.get("mesh", {})
                if mesh_cfg.get("projection", "orthographic") != "orthographic":
                    raise ValueError("Only orthographic mesh projection is supported")
                depth = mesh_to_depth_map(
                    cloud.points,
                    size=grid_sz,
                    crop_radius_ratio=float(mesh_cfg.get("crop_radius_ratio", 0.85)),
                    preserve_missing=True,
                )
            else:
                depth = np.load(sample.path_3d).astype(np.float32)
            cfg_key = key3d
            source_hash = pcache.file_digest(sample.path_2d if pseudo else sample.path_3d)
            in_digest = digest(
                [
                    source_hash,
                    pcache.file_digest(sample.path_landmarks) if sample.path_landmarks else "",
                ]
            )
            out_npy = _out_3d(interim, subject, sample_id, "_r64.npy")
            out_norm = _out_3d(interim, subject, sample_id, "_n64.npy")
            out_curv = _out_3d(interim, subject, sample_id, "_c64.npy")
            out_valid = _out_3d(interim, subject, sample_id, "_valid64.npy")
            outputs = [out_npy, out_norm, out_curv, out_valid]
            if sample.path_landmarks or pseudo:
                outputs.append(_out_lmk(interim, subject, sample_id))
            if row["preprocess_3d_hash"] != pcache.cfg_hash(cfg_key) or not all(
                pcache.is_cached(p, cfg_key, in_digest) for p in outputs
            ):
                rimg, ok, hole = preprocess_3d_sample(depth, cfg3d)
                save_array(out_npy, rimg)
                size = rimg.shape[0]
                spacing = 2.0 / max(size - 1, 1)
                normals = normals_from_depth(rimg, spacing=spacing)
                curv = curvature_from_depth(rimg, spacing=spacing)
                out_norm = _out_3d(interim, subject, sample_id, "_n64.npy")
                out_curv = _out_3d(interim, subject, sample_id, "_c64.npy")
                save_array(out_norm, normals)
                save_array(out_curv, curv)
                valid = cv2.resize(
                    np.isfinite(depth).astype(np.uint8),
                    (size, size),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                save_array(out_valid, valid)
                lms = adapter.load_landmarks(sample)
                if pseudo:
                    save_array(_out_lmk(interim, subject, sample_id), lmk3d.astype(np.float32))
                elif (
                    lms is not None and np.asarray(lms).ndim == 2 and np.asarray(lms).shape[1] == 2
                ):
                    ll = np.asarray(lms, dtype=np.float32).copy()
                    h, w = depth.shape
                    yy = np.clip(np.rint(ll[:, 1]).astype(int), 0, h - 1)
                    xx = np.clip(np.rint(ll[:, 0]).astype(int), 0, w - 1)
                    lmk3d = np.column_stack(
                        [
                            (ll[:, 0] - w / 2) / max(w, 1),
                            (ll[:, 1] - h / 2) / max(h, 1),
                            np.nan_to_num(depth[yy, xx], nan=0.0),
                        ]
                    )
                    save_array(_out_lmk(interim, subject, sample_id), lmk3d.astype(np.float32))
                for output in outputs:
                    pcache.mark_cached(output, cfg_key, in_digest)
                manifest.at[i, "nosetip_ok"] = ok
                manifest.at[i, "hole_ratio"] = hole
                manifest.at[i, "reject_reason_3d"] = "" if ok else "insufficient_geometry"
                log.debug("%s 3d ok=%s hole=%.3f", sample_id, ok, hole)
            manifest.at[i, "preprocess_3d_hash"] = pcache.cfg_hash(cfg_key)
            manifest.at[i, "content_hash_3d"] = source_hash

    if modality == "2d":
        manifest["quality_flag"] = np.where(manifest["detect_ok"], "", "rejected")
    elif modality == "3d":
        manifest["quality_flag"] = np.where(manifest["nosetip_ok"], "", "rejected")
    else:
        manifest["quality_flag"] = np.where(
            (~manifest["has_2d"] | manifest["detect_ok"])
            & (~manifest["has_3d"] | manifest["nosetip_ok"]),
            "",
            "rejected",
        )
    reject = int((manifest["quality_flag"] == "rejected").sum())
    write_manifest(manifest, manifest_path)
    log.info(
        "Preprocessing done: rejected=%d / %d rows",
        reject,
        len(manifest),
    )
    return manifest
