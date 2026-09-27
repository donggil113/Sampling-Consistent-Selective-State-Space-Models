# Research packet: Sampling-grid dependence in selective SSMs (P1)

Last updated: 2026-09-27 (third session). Branch: `claude/intelligent-maxwell-t1lcda`.

Runs covered:

- FIRST_RUN: pre-registered at `e8d127d`, amendments A1–A3 at `f88b8b9`, executed at `85b1ffe`.
- P1-COMP-01: pre-registered at `78bdcf0`, executed at `ca90992`.
- P1-REAL-01: design at `8f12ebf`; pre-run amendments R-A0..R-A9 at `2869c74`; trained at `2869c74`; evaluated at `6a2f793`. This is a **seed-0 conditional pilot**.

Manuscript: `paper/main.tex` v2, built as `paper/main.pdf` (v1 is at commit `8fb0368`). Claims map: `paper/claims.csv` (40 claims).

## 00. Revision log (third session)

1. **TIDES interval convention (paper vs. code).**
   - The paper's Eq. 2/9 holds u_k forward on [t_k, t_{k+1}), and the state inside y_k excludes u_k.
   - The pinned code holds u_k backward on (t_{k−1}, t_k] and includes it. Its collate builds steps as [t_0, diff(t)].
   - Executed check (untrained weights, fixed impulse input; `results/raw/P1-REAL-01__convention_check.json`):
     - the code equals the backward-hold recurrence to 6.5e-19;
     - on a uniform grid, a one-index shift of the state reproduces the paper form to 8.7e-19;
     - on an irregular grid the residual is 3.7e-3.
   - Conclusion: this is a genuine difference in input/gap pairing, not only notation. On uniform grids it reduces to a one-step shift of the state readout.
   - The code is what produced TIDES' results. No claim is made that this affects them.
2. **Review of the two-layer proof assumptions (Prop. 4.2).**
   - The bound \|e_k\| ≤ \|bd\| U δ e^{cδ}/c needs:
     - a ≥ 0: a\|h1\| ≤ \|b\|U for a > 0, and the term is 0 for a = 0;
     - c > 0: e^{−c(τ−s)} ≤ 1 and the sum bound e^{cδ}/c;
     - zero initial state: a nonzero h1(0) adds a\|h1(0)\| to 2\|b\|U, while h2(0) does not enter;
     - \|u\| ≤ U.
   - The Lipschitz step is sound: h1 is C¹ on each held interval, and the bracket lies within one interval.
   - As c → 0 the bound is no longer uniform in T; it grows linearly. a < 0 is not covered.
   - The stored P1-COMP-01 errors are 9–20% of the bound. This is consistency, not proof.
   - Scope is limited to the LTI scalar cascade. It gives **no guarantee** for selective coefficients, nonlinear inter-layer maps, or trained deep models.
3. **P1-REAL-01 executed as a seed-0 conditional pilot** on official TIDES (§9). The new-architecture STOP is unchanged.

## 0. Revision log (second session)

Five interpretation corrections. The earlier wording is quoted so that the change is visible.

1. **Assumptions of single-layer invariance made explicit (C0).** Exact-ZOH split invariance holds under four conditions:
   - (i) the held value is repeated on every sub-step;
   - (ii) every coefficient (g, B, C, λ) depends only on the held value;
   - (iii) the step is Δ = sub-step length × g;
   - (iv) exact ZOH formulas are used.

   Anything else feeding a coefficient breaks (ii): token-indexed conv1d, time or position features, state-dependent transitions, or time-axis normalization statistics in training mode. This is stated as Prop. 4.1 in the manuscript.
2. **Internal resampling is not new information (C8).** Earlier wording: "splitting at the input becomes *new observations* for layer 2". Corrected: layer 2 receives additional *internal* samples of layer 1's output. The resulting change is a reconstruction error of an internal signal; no external observation is added. P1-COMP-01 now checks this in a linear cascade.
3. **Bilinear: stability, oscillation and stiff-decay accuracy are separated (C3).** Earlier wording: "fails badly on stiff modes". Corrected:
   - Bilinear is stable: |Ā| < 1 for Re z < 0, at every step.
   - It is sign-alternating for real z < −2 (oscillatory transients).
   - It is not L-stable: Ā → −1 instead of e^z → 0, so stiff decay is inaccurate.
   - Its steady state under a held input is exact.

   See `paper/generated/tab_stiff.tex`. The stable/sign/decay columns there are an EXPLORATORY re-reading of logged values.
4. **Path/artifact shares replaced by an exact decomposition (C4).** Earlier wording: "artifact share 0.000 / 82–89% of the change is artifact". That column was a ratio of norms, ‖R‖/(‖P‖+‖R‖). Norms are not additive, so it is not a share. Corrected:
   - The vector identity is exact: T = P + R. Here P = C_m − C_1 is the change of the continuous-time solution on the held path, and R is the change of the discretization artifact.
   - Hence ‖T‖² = ‖P‖² + ‖R‖² + 2⟨P,R⟩. The three terms are reported as fractions of Σ_units‖T‖², which sum to 1.
   - The cross term is large and of both signs: per-unit range −1.37 to +0.42. So no per-unit "share" is well defined.
   - This decomposition is EXPLORATORY (re-aggregation of stored outputs; `results/paper_assets.json`, `paper/generated/tab_newobs.tex`).
   - The old column is kept in `results/first_run_summary.md` but relabelled "norm ratio (not a share)". Its numbers are unchanged.
5. **Endpoint sums vs. exact integral (C5).**
   - `time_riemann` is a right-endpoint time-weighted sum.
   - `time_trapz` is a trapezoidal endpoint sum.
   - `time_exact` is the exact integral of the hold-consistent continuous output. It is defined only for exact-ZOH states.
   - Only `time_exact` is split-invariant. The endpoint sums converge only where the grid is refined.

Also: the stop condition S2 is an **operational decision rule**. It is not a statistical test, an equivalence result, or a novelty argument.

## 1. Question and scope

**Question.** When only the time grid of the same physical input path is changed, where in a selective SSM does the output change enter, and which parts are artifacts?

**Scope of the evidence so far.**

- Controlled toys have fixed parameters: a single-layer selective diagonal model (FIRST_RUN) and a two-layer linear cascade (P1-COMP-01). The only trained model is official TIDES in the P1-REAL-01 seed-0 pilot (§9).
- **Nothing here is evidence about trained Mamba, S4 or S5. The TIDES evidence is one seed on one synthetic task.**
- The independent unit is one seed, which fixes one parameter draw and one input path. Grid variants of the same seed are not independent samples.
- Scenarios whose results were already seen are EXPLORATORY for any new analysis.

## 2. Claims ledger

The full list, with evidence files, is in `paper/claims.csv` (31 claims). Status vocabulary:

- `STANDARD`: known, not a contribution.
- `ENGINEERING_PASS`.
- `TOY_MEASURED`.
- `EXPLORATORY`.
- `PROVED (elementary; not independently checked)`.
- `UNPROVED`.
- `CONJECTURE`.
- `CODE_READ`.
- `PREDICTION`.
- `NOT_RUN`.
- `FAIL`.
- `WEAK_PASS`.
- `OPERATIONAL_DECISION`.
- `PRIOR_WORK`.
- `ABSENCE_OF_EVIDENCE`.

| ID | Claim (short) | Status | Evidence |
|---|---|---|---|
| C0 | Exact ZOH with Δ=dt·g is split-invariant **under assumptions (i)–(iv)** | STANDARD + ENGINEERING_PASS | H1 max 6.3e-15; RK4 agreement 1.6e-13 |
| C1 | Δ=τg gives an artifact that does not vanish, and grows with m | TOY_MEASURED, by construction | FR-E1 medians 0.48 / 0.93 / 1.28 (m=2/4/8); H4 PASS; seed 25 non-monotone for eulerB_nodt |
| C2 | Euler-B with the correct step has a first-order artifact | TOY_MEASURED | slope −1.03 (H2); median base-step artifact 0.21; magnitude depends on the toy's Δ·\|λ\|; **not measured for real Mamba** |
| C3 | Bilinear is second order, stable, sign-alternating for z<−2, and inaccurate for stiff decay; its steady state is exact | TOY_MEASURED (+ EXPLORATORY columns) | slope −2.02 (H3); FR-E4-N2 Ā=−0.977 at λ=−500; N3 |
| C4 | With new observations, the exact-ZOH change equals the change of the CT solution on the new held path | TOY_MEASURED | artifact ≤ 1.65e-13 (H5a); error to truth slope −0.98 (H5b) |
| C4b | Exact squared-norm decomposition for the other variants | EXPLORATORY | Δ=τg, exact ZOH, m=8: ‖R‖² 1.22, cross −0.25 of Σ‖T‖²; Euler-B dt: ‖R‖² 0.53, cross +0.27 |
| C5 | Sample-count readouts change under a density change on a split-invariant backbone; the exact held-path integral does not; endpoint sums converge only where the grid is refined | TOY_MEASURED | FR-E3(a), m=8: sample mean 0.30, sample sum 1.5, trapezoid 0.011, right-endpoint 0.026, exact 1.5e-16 (H7a, H7b); H8 ratio 9.0 |
| C5f | H7c_original | **FAIL** (mis-specified pre-run; amendment A2) | slope +0.42 |
| C5w | H7c_restated | **WEAK_PASS** | median slope −0.076; 21/32 negative |
| C6 | Fixes are not additive; each alone leaves at least half; only the three together remove the artifact | TOY_MEASURED | 0.46 → 0.25 / 0.48 / 0.30 → 1.5e-16 |
| C7 | The float32 floor is about 1e-4 at 16k sub-steps | TOY_MEASURED (emulated) | FR-E4-N4 |
| C8 | Stacks: the layerwise endpoint hold of an internal signal changes under splitting at first order, with no new external information | STANDARD + TOY_MEASURED (linear cascade) | P1-COMP-01: exact coupled invariant 1.7e-15; layerwise error 0.19 → 0.026 (slope −0.95; −0.93 for a=c) |
| C8p | Layerwise error bound \|e_k\| ≤ \|bd\| U δ e^{cδ}/c, uniform in T, including a=c, **for a≥0, c>0, zero initial state** | PROVED (elementary; assumptions reviewed in session 3; not independently checked) | raw errors are 9–20% of the bound (consistency only); LTI scalar cascade only |
| C9 | A token-indexed conv1d violates assumption (ii) | ANALYTIC_ARGUMENT; magnitude NOT_RUN | code fact: `d_conv=4` |
| C10 | Trained-model results | **PILOT_MEASURED for official TIDES (seed 0, one task)**; NOT_RUN for Mamba/S4/S5 | §9 |
| C11 | Official TIDES code facts: step ∝ gap; naive exp−1 in B̄; right-endpoint scan; BatchNorm over batch×time; conv off by default; mean-pooling classifier | CODE_READ + executed via adapter | configs/p1_real_01.json; convention/adapter checks |
| C12 | Paper vs. code interval pairing differs (§00.1) | CODE_EXECUTED (untrained) | results/raw/P1-REAL-01__convention_check.json |

**Separation of evidence types.**

- *Engineering PASS* (implementation correct): H1, H5a, H7a, COMP-H1–H3.
- *Toy results*: C1–C8.
- *Real-model results*: none (NOT_RUN).
- *External utility*: not assessed.
- *Novelty judgement*: the new-architecture claim is STOPPED. For the measurement/decomposition framing we found no prior controlled refinement study in the sources read; this is ABSENCE_OF_EVIDENCE, not proof.

## 3. FIRST_RUN results (unchanged; primary config: uniform base grid, real modes, 32 units)

Full tables: `results/first_run_summary.md`. Manuscript tables regenerated from raw: `paper/generated/`.

| ID | Verdict | Note |
|---|---|---|
| H1 | PASS | max 6.3e-15 ≤ 1e-12 |
| H2 | PASS | median slope −1.03 |
| H3 | PASS | median slope −2.02 |
| H4 | PASS | zoh_nodt 32/32 and eulerB_nodt 31/32 non-decreasing |
| H5a | PASS | max artifact 1.65e-13 |
| H5b | PASS | median slope −0.98 |
| H6 | PASS | medians 1.26 / 1.46 |
| H7a | PASS | max 5.3e-15 |
| H7b | PASS | holds at each m |
| H7c_original | **FAIL** | mis-specified before the run (A2); slope +0.42 |
| H7c_restated | PASS, **weak** | median slope −0.076; 21/32 negative; cause (unrefined half) not tested |
| H8 | PASS | ratio 9.0 |

## 4. P1-COMP-01 (two-layer linear cascade; STANDARD-PROPERTY CHECK)

- *System.* ḣ1 = −a h1 + b u and ḣ2 = −c h2 + d h1.
- *Cases.* (a,b,c,d) = (1,1,0.5,1) and (1,1,1,1). The second is a=c, a Jordan block.
- *Input.* The held input of FIRST_RUN unit 0 on the FIRST_RUN base grid. No new seed or grid.
- *Splitting.* m ∈ {1,2,4,8}; physical time and the original input are preserved.
- *Results.* All COMP-H1..H5 PASS for both cases. Wall clock 0.108 s.
  - The exact coupled solution is invariant to 1.7e-15 and matches RK4 to 6.1e-15.
  - Layer 1 of the layerwise computation is identical to the exact solution.
  - The layerwise layer-2 error is 0.19 / 0.10 / 0.052 / 0.026 (a≠c) and 0.20 / 0.11 / 0.055 / 0.028 (a=c).
- *Interpretation.* Every layer satisfies the single-layer conditions. The change comes only from resampling the internal signal. This is not new information and not a new architecture.

## 5. Proof candidates

Numerical agreement is not a proof.

- **P-A (STANDARD; proof in paper Prop. 4.1).** Split invariance under (i)–(iv).
- **P-cascade (PROVED, elementary; paper Prop. 4.2; not independently checked).**
  - Statement: the layerwise endpoint-hold error satisfies \|e_k\| ≤ \|bd\| U δ e^{cδ}/c.
  - Proof steps:
    - The local term is bounded by \|bd\| U τ², since \|ḣ1\| ≤ 2\|b\|U.
    - Unroll the recursion with the contraction e^{−cτ}.
    - Bound the resulting sum by e^{cδ}/c.
- **P-B (UNPROVED sketch).** The hold error is first order, uniformly in T, under A1–A4 (paper App. C). Missing: Carathéodory treatment.
- **P-C (UNPROVED sketch).** The Euler-B global error is O(δ), with a constant that grows with Δ\|λ\|.
- **P-D (CONJECTURE).** Under refinement, the Δ=τg artifact tends to a nonzero quasi-steady-state limit.
- **P-E (was CONJECTURE).** For the linear cascade it is now P-cascade. For trained selective stacks its magnitude is NOT_RUN.

## 6. Stop conditions (preserved)

| Condition | Result |
|---|---|
| S0 budget (FIRST_RUN 120 s) | not triggered (20.6 s) |
| S1 implementation bug | not triggered |
| S2 research stop | **TRIGGERED at toy level. ARCHITECTURE_CLAIM = STOPPED.** Operational rule, not a statistical or equivalence test |
| S3 no sweep | respected; P1-COMP-01 used no new seed or grid |

TIDES (arXiv 2605.09742) already occupies the design of physical Δ + exact ZOH + selectivity on Re(Λ), B, C. The manuscript is framed as measurement and decomposition only.

## 7. P1-REAL-01 design (session 2; EXECUTED in session 3 as a seed-0 pilot, see §9)

`configs/p1_real_01.json` supersedes the NX1 plan in part. The NX1 30-minute budget for all baselines was never measured and is not confirmed.

- *Implementation, fixed before results.* Official TIDES PyTorch @ `4b51adce2060e7209e002a6a2fd6691a2f6fcc5e` (MIT).
  - Selected because it is official, CPU-capable with a pure-PyTorch scan, and handles per-step physical steps.
  - Rejected: Mamba (CUDA/Triton kernels; no physical time), S5 (jax 0.3.5 pin; per-step Δ on a separate branch), and the Fading Flash notebook (JAX; absent).
- *Comparison.* The same trained backbone, fed (a) on the native grid with physical steps and (b) resampled onto the training grid.
  - Resampling: LOCF (causal, primary) or linear interpolation (non-causal, secondary).
  - Conditions: splitting, nested new observations, random times.
- *Protocol.*
  - Trajectory IDs are split first (512/128/256); grids are built afterwards within each split.
  - Labels are taken at fixed physical query times over a fixed physical horizon.
  - Normalization uses train-split statistics only.
- *Metrics.* Discrepancy, task loss and inference cost, with bootstrap CIs.
  - Equivalence is claimed only if the CI lies within ±5%.
  - Non-significance is not equivalence.
- *Approval items A-R1..A-R4.*
  - A-R1: install torch (CPU).
  - A-R2: fetch the pinned TIDES code.
  - A-R3: a ≤300 s timing smoke.
  - A-R4: a training cap to be set after the smoke.

## 8. Sources, licenses, versions

| Item | Kind | Source / version | License | Use |
|---|---|---|---|---|
| `src/scssm`, `scripts`, `tests` | code | this repo; CPython 3.11.15; stdlib only | repo license NOT SPECIFIED (owner decision) | executed |
| Toy data | data | synthetic from integer seeds: FIRST_RUN 0–31 (primary), 0–7 (secondary), 999 (timing smoke only); P1-COMP-01 reuses unit 0; P1-REAL-01 would use 1,000,000+id and 2,000,000+id | n/a | executed (P1-REAL-01: not) |
| state-spaces/mamba | external code | `main` @ e9594ce1c732d97440f0332fdc43170a2294dbfa; files read: `selective_scan_interface.py`, `mamba_simple.py`, `modules/mamba3.py` (imports) | Apache-2.0 (LICENSE read) | read only |
| TaylanSoydan/TIDES | external code | @ 4b51adce2060e7209e002a6a2fd6691a2f6fcc5e, cloned into `third_party/TIDES` (gitignored, not vendored) | MIT (LICENSE read) | executed via the adapter (P1-REAL-01) |
| torch / numpy / scipy | library | 2.14.0+cpu / 2.4.6 / 1.17.1 (pip, CPU wheel index; installed session 3) | BSD-style (not re-checked) | executed |
| TeX Live (Ubuntu 24.04 apt), latexmk, poppler-utils | toolchain | installed session 3 | distribution licenses (not re-checked) | PDF build only |
| P1-REAL-01 checkpoint | model artifact | `results/raw/P1-REAL-01__checkpoint.pt`, sha256 e4784897f5d507f0fa2bc29b2fb8d5fd7e7d81f8ad6498ce3a6f3fe9331bb455 | trained in this project (repo license NOT SPECIFIED) | evaluated |
| lindermanlab/S5 | external code | HEAD @ 3c18fdb6b06414da35e77b94b9cd855f6a95ef17; `pendulum` branch @ 52cc7e22d6963459ad99a8674e4d3cfb0a480008 (not cloned) | MIT (LICENSE read) | read only |
| ICML 2026 style kit | template | https://media.icml.cc/Conferences/ICML2026/Styles/icml2026.zip, sha256 8b29290f…, fetched 2026-09-26; not vendored | distributed by ICML for authors (license text not checked) | referenced by `paper/main.tex`; not compiled |
| Papers | literature | arXiv abs-page metadata in `paper/bib_provenance.json` (the arXiv API returned HTTP 406 via the proxy) | not checked | cited; see RELATED_WORK.md |
| External pretrained models / checkpoints | — | none used | — | — |

## 9. P1-REAL-01 results (seed-0 CONDITIONAL PILOT; official TIDES @ 4b51adc)

Setup:

- The model has 23,397 parameters and 2 blocks, with input-dependent Re Λ, B and C; exact ZOH; causal; no conv. It runs on CPU in float32 with 2 threads, using torch 2.14.0+cpu.
- The task is regression onto the FIRST_RUN teacher (unit-0 parameters). T = 8, K = 32. Label s.d. 2.35.
- Trajectory IDs were split first: 512 train / 128 calibration / 256 test.
- Training: Adam, lr 3e-3, 2000 steps, batch 32, 97 s in total.
- The checkpoint was selected on calibration MSE (step 1500, MSE 0.012), hashed, and frozen before any test trajectory was generated.
- Calibration MSE was unstable (up to 0.84).

| Condition | disc. native | disc. resampled | MSE native / resampled | ΔMSE [95% CI] | ±5% rule | runtime nat./res. (s) |
|---|---|---|---|---|---|---|
| C0 | 0 | 0 | 0.0167 / 0.0167 | 0 | identical inputs | 0.053 / 0.052 |
| S2 | 6.1e-3 | 0 | 0.0162 / 0.0167 | −5.1e-4 [−7.3e-4, −2.7e-4] | within margin | 0.16 / 0.05 |
| S4 | 9.2e-3 | 0 | 0.0161 / 0.0167 | −6.5e-4 [−9.9e-4, −2.9e-4] | native lower; not beyond margin | 0.37 / 0.05 |
| S8 | 0.011 | 0 | 0.0160 / 0.0167 | −6.9e-4 [−1.1e-3, −2.7e-4] | native lower; not beyond margin | 1.07 / 0.05 |
| H8 | 7.8e-3 | 0 | 0.0156 / 0.0167 | −1.1e-3 [−1.4e-3, −7.6e-4] | native lower; not beyond margin | 0.34 / 0.05 |
| J1 (secondary; held path differs) | 0.35 | 0.39 | 1.14 / 1.22 | −0.078 [−0.21, 0.062] | inconclusive | 0.05 / 0.05 |

- **Pooled readout** (label: time average; mean magnitude 1.59). On H8, the error is 0.56 for the sample-mean pooling of native outputs, 0.068 for time-weighted pooling of the same outputs (pooling-only), and 0.063 for resampled. R-P5 is supported.
- **Pre-registered predictions.**
  - R-P1 (native S8 discrepancy > 1e-4): supported.
  - R-P2 (resampled split discrepancy = 0): supported.
  - R-P4 (runtime): observed.
  - R-P5: supported.
- **Layer diagnostic (EXPLORATORY, post hoc).**
  - The encoder is unchanged.
  - Block 1: ≤ 7.2e-6 in fp32 and ≤ 1.8e-14 in fp64.
  - Block 2: 4.1e-3 to 7.2e-3 in both precisions.
  - This is consistent with the pattern of Props. 4.1 and 4.2 in one checkpoint. It is not a general proof.
- **Equivalence rule** (R-A5).
  - The unit is the per-trajectory MSE in label units.
  - The margin δ is 5% of the resampled mean MSE of the same condition, applied to the mean paired difference.
  - "No practical difference" requires the 95% bootstrap CI to lie inside ±δ.
  - The 5% value is a convention fixed before the results (≈ 2.5% in RMSE). It is not derived from any application cost.
- **Not shown by this pilot.**
  - Anything about the training algorithm across seeds.
  - Anything about TIDES benchmarks, Mamba, S4 or S5.
  - Any transfer to real sensor or clinical data.
  - The teacher belongs to the model's own continuous-time family, which may favour it.

