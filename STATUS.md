# STATUS — P1 Sampling-Consistent Selective SSMs

Last updated: 2026-09-26 (UTC). Branch: `claude/intelligent-maxwell-t1lcda`.

## State at start of this session

The repository was empty: no commits, no remote refs.

- These files did not exist: CLAUDE.md, STATUS.md, RESEARCH_PACKET.md, configs, results.
- No earlier STOP or ARCHIVE decisions or Work IDs existed, so none were changed or mapped.
- numpy, torch, scipy and pytest were not installed. No installs were approved, so all code is CPython 3.11 stdlib only (`unittest` instead of pytest).

## FIRST_RUN: DONE (then stopped, as instructed)

| Step | Commit | Note |
|---|---|---|
| Pre-registration | e8d127d | `configs/first_run.json`: hypotheses H1–H8, metrics, units, budget, stop conditions |
| Implementation + pre-run amendments A1–A3 | f88b8b9 | amendments recorded **before** any experiment ran (reasons are in the config) |
| Runner fix | 85b1ffe | the git-dirty check now runs before the output files are created |
| Execution | at 85b1ffe | 20.6 s wall clock of 120 s; CPU affinity {0,1}; clean tree |

### Deviations and incidents (all logged)

1. **Timing smoke test before the real run.** It used seed 999, which is not a pre-registered seed. Only timings were printed; metrics were not inspected.
2. **Run 1 was correct but its manifest was wrong.** Run 1 completed, but the manifest reported `git_dirty_at_start=true`. The runner itself had created `results/raw/run.log` before running `git status`.
   - The runner was fixed and the run repeated once.
   - All 10 data files (9 `.jsonl` plus `hypotheses.json`) are **byte-identical** between run 1 and run 2 (sha256 compared). Only `run.log` timings differ.
   - An intermediate attempt was killed by a shell pipe (`| head`) after its first experiment. Its partial outputs were deleted before the final run.
3. **H7c was mis-specified before the run.** Amendment A2 identified the problem before running. The original clause is still evaluated verbatim and FAILS, as predicted. The restated clause passes only weakly (see below).

### Results: engineering vs. science (details in RESEARCH_PACKET.md §2–3)

**Engineering: PASS.**

- Exact ZOH is invariant under splitting: max 6.3e-15.
- It matches an independent RK4 reference: 1.6e-13.
- The time-exact readout is invariant: 5.3e-15.
- Outputs are deterministic across the two runs.

**Pre-registered hypotheses (primary config, 32 units).**

- H1, H2, H3, H4, H5a, H5b, H6, H7a, H7b, H8: PASS.
- H7c_original: FAIL (mis-specified).
- H7c_restated: PASS but weak. Median slope −0.076; only 21 of 32 units have a negative slope.

**Scientific reading (toy only).**

- The grid-induced unnecessary change separates into five parts:
  - Δ not scaled by dt: does not vanish.
  - Euler-B discretization: O(dt).
  - Bilinear discretization: O(dt²), and not L-stable.
  - Sample-count readouts: do not vanish under a density change.
  - Riemann vs. exact time weighting.
- When new observations are added, the change of exact ZOH is 100% genuine path change.
- The fixes are not additive, and only the combination removes the artifact.

**Stop condition S2: TRIGGERED at toy level. ARCHITECTURE_CLAIM = STOPPED.**

- In a single layer, the simple fix (Δ=dt·g, exact ZOH B, time-exact readout) explains the entire artifact.
- TIDES (arXiv 2605.09742, 2026) already occupies the design of physical Δ + exact ZOH + selectivity on Λ.
- The *measurement and decomposition* question remains open only for trained, multi-layer models. That work is NOT_RUN.

## NOT_RUN / unverified

- Any trained model, including real Mamba, S4, S5 and TIDES. Toy failures are **not** evidence about real Mamba.
- The depth ≥ 2 effect (C8): CONJECTURE.
- The token conv1d effect (C9): CONJECTURE.
- Interpolation (first-order hold) stem and fixed-time-grid backbone (NX1): design only.
- Proofs P-B and P-C: UNPROVED sketches. P-D and P-E: CONJECTURE.
- Real float32 or bfloat16 kernels. Only float32 emulation was run.
- Literature: most per-paper details were read by the literature sub-agent only. The main author re-checked the TIDES abstract plus a keyword search of its full text, the Mamba-3 abstract, and the Mamba reference code with its license (see RELATED_WORK.md).

## Next (needs a decision from the owner)

NX1-STEM-BACKBONE (`configs/next_experiment.json`) is the one decision experiment. It **needs approval** to install a CPU tensor library, and a budget of 30 min on 2 CPU threads.

## File map

- `configs/first_run.json`: pre-registration and amendments.
- `configs/next_experiment.json`: NX1 design (NOT_RUN).
- `src/scssm/`: model, grids, RK4 references, metrics, experiments, hypothesis evaluation.
- `tests/test_model.py`: 13 unittest checks (numerical checks, not proofs). Run with `python3 -m unittest discover -s tests -t .`.
- `scripts/run_first_run.py`: budgeted runner. `scripts/summarize_first_run.py`: builds the tables from the raw records.
- `results/raw/`: per-unit records, `hypotheses.json`, `run.log`. `results/first_run_summary.{md,json}`: tables.
- `run_manifest.json`: git head, config and code sha256, environment, per-experiment status, output sha256.
- `RESEARCH_PACKET.md`, `RELATED_WORK.md`, `paper/skeleton.tex` (English skeleton with `\todo{ID}`).
