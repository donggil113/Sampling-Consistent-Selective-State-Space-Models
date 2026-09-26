# Research packet: Sampling-Consistent Selective State-Space Models (P1)

Last updated: 2026-09-26. The run this packet covers is FIRST_RUN. It was pre-registered in `configs/first_run.json` at commit `e8d127d`, with pre-run amendments A1–A3 at `f88b8b9`, and executed at `85b1ffe`.

## 1. Question and scope

**Question.** Change only the time grid of the same physical input path. How much of a selective SSM's output change is a grid artifact, as opposed to genuine new information? Can the artifact be decomposed into parts and removed?

**Scope of FIRST_RUN.**

- The model is a fixed-parameter toy: one scalar input channel and N=4 diagonal stable modes, with a Mamba-like gate g(u)=softplus(wu+β) shared across modes, a selective B(u) and a selective C(u). There is no training.
- Every statement in this packet is about that toy. **Nothing here is evidence about trained Mamba, S4 or S5.**
- There are no train/dev/test splits, because nothing is fitted. The independent unit is one seed, which fixes one parameter draw and one input-path draw. Grid variants of the same seed are not independent samples.

## 2. Claims ledger

Each claim has one status:

- `STANDARD`: known result, not a contribution.
- `TOY-VERIFIED`: numerically observed in this toy.
- `UNPROVED`: proof sketch only.
- `CONJECTURE`: stated but not checked.
- `NOT_RUN`.

| ID | Claim | Status | Evidence |
|---|---|---|---|
| C0 | Exact ZOH with Δ=dt·g(u) is the exact flow of the CT model on the right-held path, so splitting a held interval leaves it unchanged | STANDARD (elementary proof in §4) | H1: max rel. change 6.3e-15 over 32 units × m∈{2,4,8}; exact ZOH vs independent RK4 agree to 1.6e-13 |
| C1 | Δ not scaled by dt (`*_nodt`) gives a splitting artifact that does not vanish, and grows with m | TOY-VERIFIED, by construction | FR-E1 zoh_nodt median artifact 0.48 / 0.93 / 1.28 at m=2/4/8 (H4 PASS; 1 of 32 units non-monotone for eulerB_nodt: seed 25) |
| C2 | With the correct dt, the Euler-B input discretization (the one Mamba's reference code uses) gives a first-order splitting artifact | TOY-VERIFIED | median slope −1.03 (H2). Magnitude: median artifact 0.21 at the base step, change vs m=1 0.155 at m=8. The magnitude depends on the toy's Δ·\|λ\| scale; **not measured for real Mamba** |
| C3 | Bilinear with the correct dt gives a second-order artifact, but fails badly on stiff modes (Ā→−1) | TOY-VERIFIED | slope −2.02 (H3); FR-E4-N2: Ā=−0.977 and artifact 1.06 at λ=−500, m=1 |
| C4 | When new observations are added, the change of zoh_dt is entirely genuine path change (artifact share 0.000). For `*_nodt`, 82–89% of the change is artifact | TOY-VERIFIED | FR-E2: zoh_dt artifact ≤ 1.7e-13 (H5a); error to CT truth has slope −0.98 (H5b) |
| C5 | Sample-count readouts create unnecessary change even on a perfectly consistent backbone when the density changes. The time-exact readout does not | TOY-VERIFIED | FR-E3(a), zoh_dt, m=8: sample_mean 0.30 and sample_sum 1.48 (scale-normalized); time_trapz 1.1e-2; time_riemann 2.6e-2; time_exact 1.5e-16 (H7a, H7b). FR-E3(b): sample_mean error to truth rises from 0.06 to 0.28, and is 9.0× the time_exact error (H8) |
| C6 | The fixes interact. Exact ZOH alone does nothing without dt-scaling, and no single fix is enough | TOY-VERIFIED | §3.2 table |
| C7 | float32 sets a consistency floor far above float64 | TOY-VERIFIED (emulated float32) | FR-E4-N4: composition drift 1.1e-4 at m=16384 (float64: 5.2e-13) |
| C8 | In a stack of depth ≥ 2 on the observed grid, splitting at the input becomes *new observations* for layer 2: the finer held sampling of layer 1's continuous output changes layer 2's input path. Exact split invariance then fails at O(dt) even with all single-layer fixes | CONJECTURE (reasoned, NOT_RUN) | — |
| C9 | A token-indexed conv1d (Mamba `d_conv=4`) is a separate source of grid dependence | CONJECTURE (NOT_RUN in toy) | code fact verified: `nn.Conv1d`, `d_conv=4` |
| C10 | Any statement about trained Mamba/S4/S5 | NOT_RUN | — |

**Engineering vs. science.**

- *Engineering success.* The implementation is correct to float64 precision: H1, H5a and H7a pass. It matches an independent RK4 reference, and outputs are bit-identical across two runs.
- *Scientific support.* The toy separates the named artifact sources and shows that the simple fix removes them in a single layer. This **supports the stop condition**, not a new method. Most of the individual facts (C0, C2 order, C3 order, readout bias) are known in principle. What the toy adds is the controlled separation, and the magnitudes it reports are specific to the toy.

## 3. FIRST_RUN results (primary config: uniform base grid, real modes, 32 units)

Full tables are in `results/first_run_summary.md`. Raw per-unit records are in `results/raw/*.jsonl`. Hypothesis verdicts are in `results/raw/hypotheses.json`.

### 3.1 Pre-registered hypotheses

| ID | Verdict | Note |
|---|---|---|
| H1 | PASS | max 6.3e-15 ≤ 1e-12 |
| H2 | PASS | median slope −1.03 ∈ [−1.2, −0.8] |
| H3 | PASS | median slope −2.02 ∈ [−2.2, −1.8] |
| H4 | PASS | zoh_nodt 32/32 and eulerB_nodt 31/32 non-decreasing; medians at m=8: 1.28 / 1.44 |
| H5a | PASS | max artifact 1.65e-13 ≤ 1e-8 |
| H5b | PASS | median slope −0.98 |
| H6 | PASS | medians at m=8: 1.26 / 1.46 |
| H7a | PASS | max 5.3e-15 |
| H7b | PASS | holds at each m ∈ {2,4,8} |
| H7c_original | **FAIL** | as predicted before the run (amendment A2): \|r_m − r_1\| grows toward a limit (slope +0.42) |
| H7c_restated | PASS, but **weak** | median slope −0.076; only 21/32 units have negative slopes (range −2.3 to +1.9). The unrefined second half is the likely cause, since its coarse Riemann error does not shrink; this cause was not tested separately |
| H8 | PASS | ratio 9.0 ≥ 5 |

### 3.2 One-fix-at-a-time decomposition

Scenario: FR-E3(a), split of the first half only, held path unchanged, m=8. Values are median scale-normalized readout changes.

| configuration | change |
|---|---|
| Mamba-style toy recipe (Δ=τg, Euler B, sample mean) | 0.462 |
| + time-weighted readout (trapz) only | 0.245 |
| + exact ZOH B only | 0.475 |
| + Δ=dt·g only | 0.305 |
| + Δ=dt·g + trapz | 0.050 |
| + Δ=dt·g + exact ZOH + trapz | 0.011 |
| + Δ=dt·g + exact ZOH + time-exact readout | 1.5e-16 |

The decomposition is **not additive**, so it should not be reported as shares that sum to 100%.

### 3.3 Secondary configurations (descriptive, 8 units each)

- Jittered base grid: same qualitative pattern. zoh_dt splitting change ≤ 2.1e-15.
- Complex modes: bilinear_dt splitting change is larger (median 0.24 at m=8, vs 0.009 for real modes), consistent with frequency warping. This mechanism was not tested separately.

### 3.4 Failure cases and numerical logs (FR-E4)

- **N1.** Naive `(exp(z)-1)/λ` loses precision: relative error 8.0e-4 at z=−1e-14, versus ≤ 8e-17 with `expm1`. In float64 with m ≤ 65536 sub-steps the effect on composition stays small (3.6e-12 vs 1.6e-12).
- **N2.** Bilinear on stiff modes is not L-stable: Ā ≈ −0.98 and the output oscillates. Euler-B's relative output error reaches 1.7e2 at λ=−500 and dt=0.5, i.e. the output is about 170× too large.
- **N3.** Euler-B steady-state error under a held input grows with the step: 0.35% at dt=0.01, 18% at dt=0.5, 457% at dt=8. ZOH and bilinear are exact at steady state.
- **N4.** Emulated float32 drifts with m: 4.3e-6 at m=256, 1.1e-4 at m=16384. This is an emulation in which every primitive is rounded and exp is correctly rounded. It does not reproduce any GPU kernel.

## 4. Proof candidates

Numerical agreement is **not** a proof. Only P-A is a complete (and standard) argument.

**Assumptions used below.**

- (A1) Re λ_n ≤ −μ < 0 for every mode.
- (A2) g : [−U,U] → [g_min, g_max] with g_min > 0, and g is L_g-Lipschitz.
- (A3) F(v) := g(v)B(v)v is L_F-Lipschitz on [−U,U], with |F| ≤ F_max.
- (A4) u : [0,T] → [−U,U] is L_u-Lipschitz.
- The grid 0=t_0<…<t_K=T has mesh δ. The right-held path is ũ(t)=u(t_k) on (t_{k−1},t_k].

**P-A (STANDARD; elementary proof).**

- *Statement.* On (t_{k−1},t_k] the held CT model is h' = g_k(λh + B_k u_k) with constant coefficients. Variation of constants gives h(t_k) = e^{g_kλ dt_k} h(t_{k−1}) + (e^{g_kλ dt_k} − 1)λ^{-1} B_k u_k, which is exactly the zoh_dt step. Splitting the interval replaces e^{a}·e^{b} by e^{a+b}, and the input terms add up accordingly (semigroup property). Hence the recurrence at the base times is unchanged.
- *Status.* Textbook fact. Not a contribution.

**P-B (UNPROVED; sketch). Hold error is first order, uniformly in T.**

- *Statement.* Under A1–A4, sup_t |h_ũ(t) − h_u(t)| ≤ (L_g|λ|H + L_F) L_u δ / (g_min μ), where H = F_max/(g_min μ).
- *Sketch.*
  - The error satisfies e' = g(ũ)λe + [(g(ũ)−g(u))λh_u + F(ũ)−F(u)].
  - The bracket is bounded by (L_g|λ|H + L_F)L_u δ.
  - The propagator satisfies |exp(∫_s^t g(ũ)λ)| ≤ e^{−g_min μ (t−s)}.
  - Integrating gives the bound. The bound on H follows from the same argument applied to h_u with h(0)=0.
  - At grid times ũ(t_k)=u(t_k), so the output error is bounded by |C(u_k)|·|e(t_k)|.
- *Missing.* A written, checked proof, including Carathéodory solutions for the discontinuous ũ.
- *Numerical consistency.* H5b slope −0.98. This is not a proof.

**P-C (UNPROVED; sketch). Euler-B error is O(δ), uniformly in T, with a constant that grows with Δ|λ|.**

- *Statement.* The local error per step is |ΔB u|·|φ1(Δλ) − 1| ≤ |Bu| Δ²|λ|/2 for Re(Δλ) ≤ 0.
- *Sketch.* Use e^z − 1 − z = z²∫_0^1(1−s)e^{sz}ds. Sum the local errors against the contraction factor e^{−g_min μ dt}. This gives O(δ)·(δ + 1/(g_min μ)).
- *Consistency.* FR-E4-N3 shows the steady-state gain factor Δ|λ|/(1 − e^{−Δ|λ|}) exactly.

**P-D (CONJECTURE). The `*_nodt` artifact does not vanish under refinement.**

- *Statement.* With Δ=τg fixed and step δ→0, the recurrence is the exact flow of the time-rescaled system h' = (τ/δ)g(ũ)(λh + B(ũ)ũ). This is a singular perturbation whose solution approaches the memoryless quasi-steady state −B(u)u/λ. The artifact therefore tends to ‖h_CT − h_qs‖ ≠ 0 in general.
- *Numerical consistency.* FR-E1: zoh_nodt artifact 0 → 1.28 over m = 1 → 8. This is not a proof.

**P-E (CONJECTURE). See C8 (depth ≥ 2).** It is the first candidate for a non-trivial statement, and it is NOT_RUN.

## 5. Stop conditions

| Condition | Result |
|---|---|
| S0 budget (120 s, 2 CPUs) | not triggered: 20.6 s, affinity {0,1} |
| S1 implementation bug | not triggered: H1 and H5a pass |
| S2 research stop | **TRIGGERED at toy level.** The simple time handling (Δ=dt·g, exact ZOH B, time-exact readout) removes the splitting artifact to 1.5e-16 and the new-observation artifact to reference precision (≤1.7e-13) |
| S3 no sweep | respected. The run was repeated once only to fix a manifest bookkeeping bug; all data outputs were byte-identical (see STATUS.md) |

**Decision.** The **new-architecture claim is stopped** at toy level, for two reasons:

1. The simple fix explains the entire single-layer artifact.
2. TIDES (arXiv 2605.09742) already occupies the design of physical Δ + exact ZOH + selectivity on Λ.

The *measurement and decomposition* question is still open, but only for trained, multi-layer models (C8, C9). Toy failures must not be reported as failures of real Mamba.

## 6. Next decision experiment (DESIGN ONLY, NOT_RUN)

`configs/next_experiment.json` (NX1-STEM-BACKBONE) defines the experiment:

- *Factors.* Stem {linear interpolation (exact first-order hold), exact ZOH} × backbone {observed grid, fixed time grid}.
- *Required baselines.*
  - S5 (per-step Δt ZOH)
  - S4D-ZOH on the resampled fixed grid
  - **resampling + the same selective backbone**
  - the Mamba-style recipe
  - Mamba-style + Δt as an input channel
  - the simple-fix selective model
  - a TIDES-like model
- *Data.* Synthetic CT teachers. Trajectory IDs are split first (512/128/256), and grids are built afterwards within each split.
- *Resources.* It is blocked on approval to install a CPU tensor library. Budget: 30 min on 2 CPU threads.
- *Stop rules.* If either simple fix reaches ≤1e-3 splitting inconsistency at m=8 with accuracy within 5% of the best model, any architecture claim is dropped for good. If no trained model shows inconsistency above the float32 floor, the question itself is dropped.

## 7. Sources, licenses, versions

| Item | Kind | Source / version | License | Use |
|---|---|---|---|---|
| `src/scssm`, `scripts`, `tests` | code | this repo, commits e8d127d..85b1ffe; CPython 3.11.15, stdlib only | repo license NOT SPECIFIED (owner decision) | executed |
| Input paths, parameters | data | synthetic, generated from integer seeds 0–31 (primary) and 0–7 (secondary); seed 999 used only for a timing smoke test whose metrics were not inspected | n/a | executed |
| state-spaces/mamba | external code | `main` @ e9594ce1c732d97440f0332fdc43170a2294dbfa (fetched 2026-09-26); `selective_scan_interface.py` sha256 a570f4f1…, `mamba_simple.py` sha256 a17e4c51… | Apache-2.0 (LICENSE header read) | read only, to check the discretization; not vendored, not executed |
| Papers in RELATED_WORK.md | literature | arXiv versions current at fetch time (not pinned) | not checked | read (see access levels) |
| Pretrained models | — | none | — | — |
