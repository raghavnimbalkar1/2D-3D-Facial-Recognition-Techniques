# Implementation and acceptance status

Current implementation: 2026-10-01. This document supersedes the historical
progress percentages, in-flight batch claims and completion assertions in
the earlier handoff, presentation guide and result preamble.

## Local validation evidence (2026-10-01)

Validated on Windows with Python 3.12.14:

- `python -m pytest --junitxml=.venv/test-results-final.xml`: **124 passed**,
  **79% statement coverage**, including all end-to-end tests (403.46 seconds).
  Five scikit-image warnings concern LBP comparisons on normalized floating
  point inputs; the existing descriptor behavior is retained.
- All **49 dependency pins** match the installed environment;
  `python -m pip check` reports no broken requirements.
- The critical Ruff checks (`E9,F821,F822,F823`) and `git diff --check` pass.
- The CLI generated and preprocessed 48 synthetic observations from four
  subjects, generated **20 modality-specific splits**, and completed all
  **20 E00 runs**. The leakage audit reports **0 violations**. The full-matrix
  audit accepts all 20 against the current source and inputs. Re-running the
  CLI reuses those bundles without adding duplicates.
- [Generated synthetic report](../results/toy-current/RESULTS.md) and
  [audit record](../results/toy-current/validation_E00.json) are available
  locally under the ignored results directory. Verification operating points
  are exported in `results/toy-current/tables/T3_verification.csv`.
- E11 certification correctly exits nonzero because the Tufts manifest/data
  is absent. No new real-data accuracy or completed E11 claim is made.

The first remote run of commit `23c8cde` passed on five platform/Python
combinations. Windows/Python 3.12 stopped at `pip check` because the runner's
preinstalled `pipx` required newer `packaging` and `platformdirs` versions
than the project lockfile. No tests ran on that job. The 2026-10-02 workflow
correction creates an isolated virtual environment for every matrix entry,
keeps the lockfile unchanged, uses Node 24 actions and pins Ubuntu to 24.04.
The corrected run's final outcome is available in GitHub Actions.

## Architecture

`ConfigResolver` merges base, dataset, preprocessing and feature/matcher YAML
defaults into a frozen experiment configuration. Dataset adapters discover
captures; ingestion records their source hashes and cohort provenance in a
manifest. Preprocessing writes atomic content-checked bundles and records
quality independently for 2D and 3D. Eligible samples enter deterministic
modality-specific split files. The runner consumes those exact files, fits
features on the permitted training IDs, scores gallery/probes and verification
pairs, then writes metrics, figures, tables, provenance and a checksummed
completion record. Aggregation accepts only complete, compatible, unique runs.

```text
configs -> resolved experiment
raw -> ingestion manifest -> preprocessing bundles -> quality eligibility
  -> persisted splits -> training-only feature fit -> similarity matching
  -> metrics/figures/tables -> completed.json -> full-matrix audit -> report
```

## Delivered engineering changes

| Milestone | Implemented outcome | Acceptance command / check |
|---|---|---|
| Environment/configuration | Pinned Python 3.10–3.12 environment; merged and validated component defaults; explicit unsupported capability failures | `python -m pip check`; `python -m pytest tests/test_config.py tests/test_cli.py` |
| Input integrity | Raw-content hashing, atomic cache bundles, producer-code invalidation, photo-only Tufts discovery, explicit quality decisions | `python -m pytest tests/test_cache.py tests/test_tufts3d.py tests/test_pipeline_integrity.py` |
| Protocol integrity | Saved splits are mandatory; training membership, pair labels, cohort coverage, capture/content duplication and stale manifests checked | `python -m pytest tests/test_splits.py tests/test_leakage_audit.py tests/test_protocol_eligibility.py`; `python scripts/verify_no_leakage.py --data-root data` |
| Features and metrics | Correct PCA leading-component removal; tie-safe empirical FAR budgets; FAR resolution and achieved FAR; identity-cluster EER intervals | `python -m pytest tests/test_features.py tests/test_metrics.py tests/test_verification_operating_points.py` |
| Timing and robustness | Warmed repeated fit/transform/match measurements; inference time excludes plots/IO; separate clean and degraded probe artifacts; E14 | `python -m pytest tests/test_timing.py tests/test_robustness.py tests/test_pipeline_integrity.py` |
| Geometry validity | ASCII PLY validation and point counts; measured missing support; valid masks; normalized-grid differential geometry; independent-capture gate | `python -m pytest tests/test_geometry.py tests/test_preprocess_3d.py tests/test_tufts3d.py tests/test_protocol_eligibility.py` |
| Run/report validity | Resume checks config/splits/inputs/code/dependencies and artifact digests; duplicate/stale/incomplete runs fail; reports derive completion from artifacts | `python -m pytest tests/test_aggregate.py tests/test_pipeline_integrity.py`; `python scripts/verify_experiment.py --exp E00 --data-root data --results-root results/toy-current` |
| Continuous integration | Linux and Windows, Python 3.10/3.11/3.12; full pytest including synthetic end-to-end tests | `.github/workflows/tests.yml`; `python -m pytest` |

Tests intentionally use synthetic data and temporary directories. The all-method
integration fixture uses 20 bootstrap resamples for runtime; research configs
default to 1,000. The original full E00 test retains the research default.
CI configuration is supplied; remote CI success must be observed after pushing.

## Real-data gates still open

1. **E11 recertification:** the checkout contains the historical summary but no
   Tufts raw data or 40 original run bundles. Supply the licensed photos, ingest,
   preprocess, generate 10 new 2D splits and execute all 40 runs. The current
   audit must pass before calling the 2D benchmark complete. Photo-only input
   can change the cohort relative to the historical mesh-matched 110 subjects.
2. **E14 robustness:** execute its 20 P2 runs (four methods, five seeds); every
   run contains clean plus three controlled occlusion conditions. Natural
   expression/sunglasses conditions remain separately reported. Audit E14.
3. **E12/E13:** obtain independently acquired repeated 3D captures per subject,
   implement/validate an adapter for that source, verify orientation and unit
   conventions, and match subject/condition capture counts between modalities.
   Tufts TD_3D's single mesh per subject cannot satisfy the current protocol.
   Same-mesh rotations, synthetic noise or copied files are not independent
   real acquisitions and cannot support a real recognition claim.

## Reproduction (PowerShell)

Run setup from README, then from the repository root:

```powershell
.venv/Scripts/ivafr.exe dataset-build --name toy --data-root data --n-subjects 4 --n-samples 12 --size 120 --seed 7
.venv/Scripts/ivafr.exe ingest --dataset toy --data-root data
.venv/Scripts/ivafr.exe preprocess --dataset toy --data-root data --modality both
.venv/Scripts/ivafr.exe splits --dataset toy --data-root data --modality both
.venv/Scripts/ivafr.exe run --exp E00 --data-root data --results-root results/toy-current
.venv/Scripts/python.exe scripts/verify_no_leakage.py --data-root data
.venv/Scripts/python.exe scripts/verify_experiment.py --exp E00 --data-root data --results-root results/toy-current
.venv/Scripts/ivafr.exe aggregate --results-root results/toy-current --out results/toy-current
```

For real photos, after extracting `TD_RGB_E/<subject>/...` under `data/raw`:

```powershell
.venv/Scripts/ivafr.exe ingest --dataset tufts3d --data-root data
.venv/Scripts/ivafr.exe preprocess --dataset tufts3d --data-root data --modality 2d
.venv/Scripts/ivafr.exe splits --dataset tufts3d --data-root data --modality 2d
.venv/Scripts/ivafr.exe run --exp E11 --exp E14 --data-root data --results-root results/tufts-current
.venv/Scripts/python.exe scripts/verify_no_leakage.py --data-root data
.venv/Scripts/python.exe scripts/verify_experiment.py --exp E11 --data-root data --results-root results/tufts-current
.venv/Scripts/python.exe scripts/verify_experiment.py --exp E14 --data-root data --results-root results/tufts-current
.venv/Scripts/ivafr.exe robustness --exp E14 --results-root results/tufts-current
.venv/Scripts/ivafr.exe aggregate --results-root results/tufts-current --out results/tufts-current
```

Success means every command exits zero, the E11 audit reports 40 unique runs,
the E14 audit reports 20, all configured metrics/artifacts exist, and the
generated report records both complete matrices and their input audits.
Re-run the audit after changing inputs; a stored audit records the state at
its execution time, not continuous monitoring.

## Interpretation boundaries

- P1 selects one canonical gallery sample per subject. Its seeds vary sampled
  verification pairs, not independent identification galleries. Do not treat
  five identical P1 identification scores as five independent experiments.
- Verification thresholds are descriptive evaluation ROC operating points,
  **not calibrated deployment thresholds**. Report achieved FAR and pair-count
  resolution; zero empirical false accepts does not establish a population FAR.
- Confidence intervals resample subject clusters. Small evaluation cohorts,
  especially toy subjects, do not justify claims about real people.
- 3D maps use normalized grid coordinates and normalized depth. Normal and
  curvature channels are descriptors, not physical measurements in millimetres.
  Orthographic projection assumes the source mesh is already oriented with x/y
  in the face plane and z as depth; automatic real-mesh pose alignment is not
  certified. Missing support is measured before nearest-neighbor filling.
  The legacy manifest column `nosetip_ok` currently acts as a range-quality
  flag; it is not evidence of anatomical nose-tip detection.
- `--anonymize` pseudonymizes subject IDs only. Face images, sample IDs and
  original paths can remain identifying. Do not publish raw data/run metadata
  containing restricted source paths as anonymized data.
- Fusion, learned classifiers, ArcFace and unsupported experiment placeholders
  are not advertised as completed benchmark capabilities.
