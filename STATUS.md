# STATUS — P1 Sampling-grid dependence in selective SSMs

Last updated: 2026-09-27 (UTC), session 4. Branch: `claude/intelligent-maxwell-t1lcda`.

## Standing decisions (preserved)

- **ARCHITECTURE_CLAIM = STOPPED.** S2 was triggered at toy level; TIDES is prior art. S2 is an operational rule, not a statistical, equivalence or novelty test.
- **H7c_original = FAIL. H7c_restated = WEAK_PASS.**
- **Trained evidence.**
  - Official TIDES only, on one synthetic task.
  - Seed 0 is the development pilot (P1-REAL-01).
  - Seeds 1–4 are the fixed-protocol repetition (P1-REAL-02).
  - Mamba, S4 and S5 training is NOT_RUN.
- **EXPLORATORY items:**
  - the FR-E2 decomposition;
  - the stiffness columns;
  - the development-checkpoint layer diagnostic;
  - the development run's fixed-estimated-margin analysis.
- **P1-REAL-02 primary estimand** (H8 pooled-readout effect) was chosen after seeing the development run. It is not a blind choice.

## Session 4: what was done

**Authorization, stated precisely.**

- This session's user instruction explicitly requests P1-REAL-02 with training seeds 1–4 and a v3 PDF. It states that it is *not* an approval for new installs, data or weight imports, GPU, paid APIs or remote publication.
- Only already-installed software was used: torch 2.14.0+cpu, numpy, scipy and TeX Live were installed in session 3, together with the pinned TIDES clone.
- The session-3 install itself rested on Claude's *interpretation* (R-A0) of a user instruction. It was not a separate approval block.
- Push to the designated branch follows the environment's standing instruction for this branch.
- `/nvmedata/...` does not exist in this sandbox, so nothing was moved or created there. `/data000` was not touched.

**Closed without training.**

- **A. Timing direction.**
  - No repository text claimed that resampled evaluation is slower. All texts and raw data agree that native is slower at S8.
  - Real defect: the development-run "runtime" was a forward-only model call on prebuilt tensors, outside inference mode and without warm-up, and it excluded resampling. It is now labelled so.
  - P1-REAL-02 measures end-to-end and forward-only time under a fixed protocol. Ratios are named explicitly, e.g. native_S8_end_to_end / resampled_S8_end_to_end = 9.7–11.5.
  - On J1 (32 tokens each), the development forward times were 0.049 s native vs 0.053 s resampled, i.e. noise.
- **B. Pooled quantities.**
  - 0.56 / 0.068 / 0.063 (development, H8) are mean absolute errors of pooled predictions against the pooled label ȳ = T⁻¹∫y dt, in label units. They are not hidden-state discrepancies and not MSEs.
  - token-mean = p_mean = L⁻¹Σŷ_k.
  - pooling-only = p_time = T⁻¹Σŷ_k dt_k. This is a readout ablation outside the official model and not an exact integral.
  - resampled = p_res = K⁻¹Σŷ^res_k over the K = 32 resampled tokens. On H8 it equals the C0 pooled prediction, so 0.063 is the C0 error.
  - The pooled label ȳ is computed by Simpson's rule on the teacher's RK4 nodes.
  - Pooled error of an arm = n⁻¹Σ_i |p_i − ȳ_i|.
  - The definitions are now in the paper §5 and in `configs/p1_real_02.json`.
- **C. Endpoint finding.** This is a difference between the pinned implementation and the hold convention stated in the paper. The check (`scripts/check_tides_convention.py`, `results/raw/P1-REAL-01__convention_check.json`) is preserved. It is not a claim about the TIDES benchmark results.

**P1-REAL-02** (`configs/p1_real_02.json`, fixed at 5a587b7 before any execution):

| Step | Wall | Result |
|---|---|---|
| train/dev data | 39.7 s (CPU 39.4 s) | sha256 dad632e9…; normalization stats exactly equal to the pilot |
| seed-0 equivalence retrain | 57.4 s | every tensor equal to the frozen pilot checkpoint; same selected step and calibration curve |
| train seeds 1–4 | 46.5 / 47.1 / 48.7 / 45.6 s (loop CPU ≈ 89–95 s each, 2 threads) | selected steps 1300 / 1600 / 800 / 1800; calibration MSE 0.0127 / 0.0135 / 0.0168 / 0.0125; calibration curves unstable in every seed |
| new test set (ids 10000–10255) | 16.1 s | sha256 e5833064… |
| eval development row | 29.9 s **INVALID** (I-1: missing output dir, crashed before writing) + 30.9 s rerun | only that row was rerun |
| eval seeds 1–4 | ≈ 30 s each | see below |
| aggregate | 0.3 s | `results/raw/P1-REAL-02/aggregate.json` |

**Results** (per seed, shared 256-trajectory test set; the crossed design is not treated as iid):

- **Primary D_s** (H8 pooled-readout effect, label units): 0.578 / 0.584 / 0.602 / 0.602. The CI excludes 0 in 4/4 seeds, so the effect is **REPLICATED WITHIN THIS TASK**.
- **S8 final-output discrepancy:** 0.0113 / 0.0219 / 0.0043 / 0.0147. It is above the floor in all seeds and equal in fp64. It varies 5.1-fold across seeds.
- **Loss contrast r_s:** **NOT STABLE.**
  - S8: −6.2% to +3.1%.
  - H8: −4.9% to +6.7%; seed 2 is higher, with a CI that excludes 0.
  - Seed 3 shows no practical difference on all held-path conditions.
- **J1** (not a refinement): r_s from −22.6% to −21.2% on the new test set in all seeds. The same development checkpoint gives −20.9% on the new test set but −6.4% (inconclusive) on its old test set, so the result depends on the test draw.
- **Conditional layer diagnostic:** not triggered, because every seed's S8 discrepancy is above 1e-3.

**Manuscript v3.**

- `paper/main.tex` (v2 at dce4d30) and `paper/main.pdf`, 12 pages.
- The main body ends on page 6 (Conclusion). Impact Statement and References follow.
- 0 undefined references or citations; 0 overfull boxes.
- PDF metadata: Author "Anonymous Authors"; Subject is the unmodified ICML 2026 style default.
- Title kept.
- `paper/claims.csv`: core claims K1–K3 plus 40 earlier rows, with superseded development claims re-labelled.
- Timing text states that end-to-end and forward-only are separate timing jobs. In 4 of 60 seed×condition×arm cells the end-to-end median is below the forward-only median, by up to 9.5%. Differences of a few percent are therefore run-to-run variation; the ~10× S8 ratio is not.
- The pooled-table caption (App. C) and §5 give every pooled quantity as a formula.
- Checks:
  - `scripts/check_paper.py` (static) now also covers the P1-REAL-02 macros.
  - New `scripts/check_pdf.py` checks the built PDF → `results/pdf_check.json`:
    - pages;
    - conclusion / impact / references / appendix start pages;
    - log warnings;
    - metadata;
    - anonymity strings in the text and metadata.

**Export bundle.**

- `export_bundle/` is built by `scripts/make_export_bundle.py`. It contains the manuscript source and PDF, configs, code, tests, small raw and aggregates, manifests and a README, with sha256 in `MANIFEST.sha256`.
- Excluded:
  - checkpoints;
  - cached synthetic data (regenerable; hashes kept);
  - third-party TIDES code;
  - the style kit and build intermediates.

## Next research decision (not run)

Decide whether an **independent real irregular time-series dataset** is needed and admissible, rather than extending this synthetic repetition.

- The decision needs the owner to name a dataset with a known license, entity-level split and sampling process.
- Importing any dataset requires explicit approval; none is assumed.

## Session 3: what was done

**Approval interpretation.** The session-3 user instruction explicitly asked to:

- verify the adapter in a real CPU tensor environment;
- train one synthetic task;
- build the PDF.

This was recorded as approval R-A0 for A-R1..A-R4 plus a TeX toolchain.

| Step | Commit | Result |
|---|---|---|
| Convention check, adapter verification, pre-run amendments R-A0..R-A9, runner | 2869c74 | TIDES code = backward-hold recurrence (6.5e-19); paper form differs (uniform: one-index shift 8.7e-19; irregular: 3.7e-3). Adapter: forward, backward and Adam update OK |
| Training (seed 0) | 6a2f793 | 97 s; checkpoint step 1500 selected on calibration (MSE 0.012), sha256 e4784897…; calibration curve unstable (max 0.84) |
| Paired evaluation + exploratory layer diagnostic | 9d48683 | see RESEARCH_PACKET §9 |
| Proof-assumption review, manuscript v2, PDF build, docs | (this commit) | Prop. 4.2 assumptions made explicit (a≥0, c>0, zero initial state, LTI scope); `paper/main.pdf` built |

**Key pilot numbers** (256 paired test trajectories; details in RESEARCH_PACKET §9):

- **Discrepancy on the same held path.**
  - Native: 6.1e-3 at S2 and 0.011 at S8, above the float32 floor.
  - Resampled: 0 by construction.
- **Task loss.**
  - Native is lower by 3.0–6.5% relative.
  - S2 is within the ±5% margin.
  - S4, S8 and H8: the CI excludes 0 but does not lie beyond the margin, so neither equivalence nor a practical difference is established.
- **Runtime.** Native goes from 0.053 s to 1.07 s per batch between C0 and S8. Resampled stays at about 0.05 s.
- **Pooled readout (H8).**
  - Sample-mean pooling errs by 0.56.
  - Time-weighted pooling of the same outputs (pooling-only) errs by 0.068.
  - Resampled errs by 0.063.
- **Layer diagnostic (EXPLORATORY).**
  - Block 1 is invariant up to rounding: 1.8e-14 in fp64.
  - Block 2 changes by 4.1e-3 to 7.2e-3 in both precisions.
- **J1** (secondary; held path differs): inconclusive.

## Compute and cost ledger, session 3 (CPU; no GPU, no paid API, no external data)

| Item | Wall clock |
|---|---|
| pip install torch 2.14.0+cpu, numpy 2.4.6, scipy 1.17.1 | 35 s |
| apt install TeX Live subset, latexmk, poppler-utils | 66 s |
| TIDES clone (pinned 4b51adc, into gitignored `third_party/`) | < 5 s |
| Convention check (3 runs; the first two were non-deterministic until numpy was seeded) | ≈ 6 s |
| Adapter check and timing smoke (2 runs) | ≈ 8 s |
| Training, including pure-Python label generation | 97.4 s |
| Evaluation | 24.5 s |
| Layer diagnostic | 11 s |
| PDF builds (several; asset regeneration included) | < 2 min total |

## Manuscript v2

- **Files.**
  - Source: `paper/main.tex` (v1 is at commit 8fb0368).
  - PDF: `paper/main.pdf`, built with the official ICML 2026 style (unmodified, review mode) by `scripts/build_paper.sh`. The kit is downloaded or copied and sha256-checked into gitignored `paper/build/`, not vendored.
  - TARGET_YEAR = 2027, TEMPLATE_YEAR = 2026, SUBMISSION_READY = false.
- **Build checks.**
  - 11 pages in total; the main body ends on page 6, within the 8-page limit.
  - 0 undefined citations or references; 0 overfull boxes after fixes.
  - No identifying strings.
  - Visual check of the rendered pages: tables and the figure are not clipped. The figure legend was moved off the data.
- **Remaining \todo markers: none.** Remaining open work (multiple seeds, other implementations, other tasks) is stated in Limitations, not as TODOs.

## Session 3 next-decision record (superseded by session 4)

The one next decision experiment was **P1-REAL-02**: the same frozen protocol with additional training seeds (for example 1–4), to test whether the pilot's discrepancy, loss and pooling findings are seed-stable.

- It needs approval of the CPU budget, estimated at about 2 min per seed from the measured pilot. This is an estimate, not yet run.
- No new task, model or implementation is involved.
- The existing seed 0 stays a pilot and must not be pooled post hoc as confirmatory evidence.

## Session 2: what was done (historical record; superseded where session 3 says otherwise)

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

## Session 2 next-decision record (superseded by session 3)

The single next decision experiment was **P1-REAL-01** (`configs/p1_real_01.json`): train official TIDES once on one small synthetic task, then compare native-grid and training-grid-resampled evaluation on the same test trajectories. It needs these unapproved resources:

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
- Session 3 additions:
  - `scripts/check_tides_convention.py`;
  - `scripts/verify_tides_adapter.py`;
  - `scripts/run_p1_real_01.py` (train and eval stages);
  - `scripts/diagnose_p1_real_01_layers.py`;
  - `scripts/build_paper.sh`;
  - `results/raw/P1-REAL-01*` (checkpoint, train, summary, per-trajectory jsonl, convention, adapter and layer checks);
  - `paper/main.pdf`;
  - `third_party/TIDES` (gitignored).

