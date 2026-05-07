# ot-sae-alignment

Project: Gromov–Wasserstein matching of sparse-autoencoder feature dictionaries across models.
Target venue: OTML @ NeurIPS 2026.

## Status

**Phase 1 in progress** — building OT primitives from scratch as the foundation for later phases.
See `docs/PROJECT_PLAN_PHASE_1.md` for the current phase plan.

## Quickstart

Create the conda environment:

    conda env create -f environment.yml
    conda activate ot-sae-alignment

Install the package in editable mode:

    pip install -e .

Install git hooks:

    make hooks

Verify the environment:

    make smoke

Run the test suite:

    make test

## Repository structure

    ot_primitives/   the OT library being built in Phase 1
    tests/           pytest test suite
    scripts/         utility scripts (smoke test, etc.)
    notebooks/       exploratory and demo notebooks
    docs/            project plan, derivations, open questions

## Reproducibility

All randomized code paths use `numpy.random.default_rng(seed)` with the seed declared
explicitly. Notebooks have outputs stripped via `nbstripout`. Environment is pinned in
`environment.yml` and `requirements.txt`.
