# STATUS — P1 Sampling-grid dependence in selective SSMs

Last updated: 2026-09-26 (UTC), session 2. Branch: `claude/intelligent-maxwell-t1lcda`.

## Standing decisions (preserved)

- **ARCHITECTURE_CLAIM = STOPPED.** Stop condition S2 was triggered at toy level in FIRST_RUN, and TIDES (arXiv 2605.09742) is prior art for physical Δ + exact ZOH + selectivity on Re(Λ), B, C. S2 is an operational rule, not a statistical, equivalence or novelty test.
- **H7c_original = FAIL.** It was mis-specified before the run (amendment A2).
- **H7c_restated = WEAK_PASS.** Median slope −0.076; only 21 of 32 units negative.
- **Trained-model results (Mamba, S4, S5, TIDES) = NOT_RUN.**
- EXPLORATORY analyses are labelled as such: the FR-E2 exact decomposition and the stiffness columns.

## Session 2: what was done

State at start:

- HEAD was `306a772`, in sync with the remote.
- No CLAUDE.md exists.
- No LaTeX compiler, numpy, torch or matplotlib is installed, and installing them is not approved.

| Step | Commit | Result |
|---|---|---|
| P1-COMP-01 pre-registration | 78bdcf0 | `configs/p1_comp_01.json` |
| P1-COMP-01 code + tests | ca90992 | `src/scssm/cascade.py`, `tests/test_cascade.py` |
| P1-COMP-01 run | 13f8927 | all COMP-H1..H5 PASS; 0.108 s; clean tree |
| P1-REAL-01 design, adapter and protocol | 8f12ebf | `configs/p1_real_01.json`, `src/scssm/real/`, `scripts/run_p1_real_01.py`; torch stages NOT_RUN |
| Paper assets, manuscript v1, claims map, doc corrections | (this commit) | see below |

**Interpretation corrections.** The details are in RESEARCH_PACKET.md §0.

1. The single-layer invariance assumptions (i)–(iv) are now explicit.
2. Internal resampling between layers is no longer called new information.
3. For bilinear, stability, oscillation and stiff-decay accuracy are separated.
4. The norm-ratio "share" is replaced by the exact decomposition T = P + R with its cross term. This is a re-aggregation of existing raw records; the old column is relabelled, not deleted, and its numbers are unchanged.
5. Endpoint time-weighted sums are distinguished from the exact held-path integral.

**Source checks (self).**

- TIDES paper: §1–3.2, §4, §5.3, §6–8, App. A–C.
- TIDES code @ 4b51adc (MIT).
- Mamba-3: §3.1 and the §4 headings.
- The official Mamba repo @ e9594ce: `mamba3.py` imports Triton.
- S5: Eq. 6, §6.2, §6.3, B.1.3 and F.3.
- S5 code @ 3c18fdb: jax 0.3.5 pinned; the pendulum code is on a separate branch.
- Keyword checks of S4, LSSL, S4D, Mamba, Mamba-2, Liquid-S4, Cirone et al., NCDE, Log-NCDE, S4ND, event-camera SSMs and LinOSS.

**Implementation fixed before any result.** The official TIDES PyTorch at commit `4b51adce2060e7209e002a6a2fd6691a2f6fcc5e`, for the native-grid vs. training-grid-resampling comparison.

## Compute and cost ledger (all CPU, 2 threads; no GPU, no paid API, no external data)

| Item | Wall clock | Note |
|---|---|---|
| FIRST_RUN timing smoke (seed 999) | ≈2.3 s | session 1; metrics not inspected |
| FIRST_RUN run 1 | 21.3 s | session 1; manifest bookkeeping bug |
| FIRST_RUN partial run (killed by pipe) | ≈0.9 s | session 1; outputs deleted |
| FIRST_RUN run 2 (reported) | 20.6 s | session 1; data outputs byte-identical to run 1 |
| P1-COMP-01 | 0.108 s | session 2 |
| P1-REAL-01 `--stage plan` | 0.001 s | protocol checks only, no model |
| Asset/summary/bib/check scripts, unit tests | < 5 s total | aggregation and tests; no experiment re-run |
| Network reads | — | arXiv HTML and abs pages, GitHub raw files and shallow clones (scratchpad, read only), ICML 2026 style kit (scratchpad, not vendored); the arXiv API returned HTTP 406 |

## Manuscript v1

- **Path.** `paper/main.tex`. Tables, figure and number macros are generated in `paper/generated/` by `scripts/make_paper_assets.py` from `results/raw`. The bibliography is `paper/references.bib`, built by `scripts/make_bib.py` from arXiv metadata. The claims map is `paper/claims.csv` (31 claims).
- **Template.**
  - TARGET_YEAR = 2027 and TEMPLATE_YEAR = 2026. No official ICML 2027 style page exists: `icml.cc/Conferences/2027*` returns 404, and icml.cc lists conferences only up to 2026.
  - The official `icml2026.sty` is used unmodified, in review/anonymous mode. It is **not vendored**; its source URL and sha256 are in the `main.tex` header.
  - SUBMISSION_READY = false.
- **Completed sections.** Abstract, Introduction, Related Work, Problem Setup, Analysis (Prop. 4.1 standard; Prop. 4.2 standard with proof), Controlled Results, Trained-models protocol (NOT RUN), Limitations, Conclusion, Impact Statement, and Appendix A–E.
- **Remaining TODOs.** Four `\todo{P1-REAL-01: ...}` markers: abstract, §5.6, conclusion, and appendix E.
- **Compile status: COMPILE_NOT_RUN.** No pdflatex, latexmk or tectonic is available.
  - The static check (`scripts/check_paper.py` → `results/paper_check.json`) found:
    - 0 undefined citations;
    - 0 undefined references;
    - 0 undefined number macros;
    - balanced environments and braces;
    - no identifying strings in `main.tex`;
    - a main body of about 3,660 words.
  - Page count, overfull boxes, table and equation clipping, and float placement are **UNVERIFIED**.
- `paper/skeleton.tex` (session 1) is kept and marked superseded.

## NOT_RUN / unverified

- Every trained-model result, and the P1-REAL-01 predictions R-P1..R-P4.
- The magnitude of the token conv1d (C9) and of normalization over the time axis.
- Stacks deeper than the two-layer linear cascade.
- Proof sketches P-B and P-C are UNPROVED; P-D is a CONJECTURE. Prop. 4.2 is proved in the paper (elementary) but not independently checked.
- Real float32 and bf16 kernels; only an emulation was run.
- S7, Δt-Mamba3D, SeRpEnt and SS-NO were read by the literature agent only, and are not cited in the manuscript.

## Next decision (needs the owner's approval)

The single next decision experiment is **P1-REAL-01** (`configs/p1_real_01.json`): train official TIDES once on one small synthetic task, then compare native-grid and training-grid-resampled evaluation on the same test trajectories. It needs these unapproved resources:

- **A-R1:** install torch (CPU, ≥ 2.4), numpy and scipy.
- **A-R2:** clone TIDES at the pinned commit into `third_party/TIDES`.
- **A-R3:** a timing smoke of ≤ 300 s.
- **A-R4:** a training wall-clock cap, set after the smoke.

## File map

- `configs/`:
  - `first_run.json`: FIRST_RUN pre-registration and amendments.
  - `p1_comp_01.json`.
  - `p1_real_01.json`.
  - `next_experiment.json`: NX1, kept, with an append-only status note.
- `src/scssm/`:
  - model, grids, reference, metrics, experiments, hypotheses;
  - `cascade.py` (P1-COMP-01);
  - `real/protocol.py` (stdlib, tested);
  - `real/tides_adapter.py` (torch; not executed).
- `scripts/`:
  - `run_first_run.py`;
  - `summarize_first_run.py`;
  - `run_p1_comp_01.py`;
  - `run_p1_real_01.py`;
  - `make_paper_assets.py`;
  - `make_bib.py`;
  - `check_paper.py`.
- `tests/`: 23 unittest checks (numerical checks, not proofs). Run with `python3 -m unittest discover -s tests -t .`.
- `results/raw/`: FIRST_RUN and P1-COMP-01 raw records, hypothesis verdicts and logs, and the P1-REAL-01 plan check.
- `results/`:
  - `first_run_summary.{md,json}`;
  - `paper_assets.json`;
  - `paper_check.json`.
- `run_manifest.json`: FIRST_RUN fields unchanged, plus `subsequent_runs` (P1-COMP-01).
- `paper/`: `main.tex`, `generated/`, `references.bib`, `bib_provenance.json`, `claims.csv`, `skeleton.tex` (superseded).
- `RESEARCH_PACKET.md`, `RELATED_WORK.md`.
