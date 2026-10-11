# STATUS — P1 Sampling-grid dependence in selective SSMs

Last updated: 2026-10-05 (UTC), session 7 (round 7). Branch: `claude/intelligent-maxwell-t1lcda`.

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

## Session 8 (round 8): what was done

**PDF delivery first.** HEAD at start a4f7aac (v5.1), clean tree; `paper/main.pdf` signature `%PDF-1.5`, 18 pages, body end page 8, sha256 567975521d89695b156134da04779480fedaabf86d34d1ab06d1e25721eee27a (matches the hash quoted by the user). Copied byte-for-byte to `deliverables/P1_v5.1_a4f7aac.pdf` with `deliverables/P1_delivery.json` and attached before any new work (commit 5211955).

**R8-P1-HAR-REPLICATION (COMPLETED; exactly the finite stage list; nothing further queued).**

- Contract `configs/auto_run_r8.json` and the minimal runner `scripts/run_p1_har_r8.py` were committed before execution (5211955); the P1-HAR-01 config hash e603cf75…, the data hashes and the original checkpoint hash e3b414f5… were verified before every stage. Seeds 101 and 102 had never been used in this project. Model RNG (python/NumPy/torch seeded before construction) and minibatch-order RNG (`random.Random(seed)`) are separate; data, splits and normalization are fixed arrays. The SAME nine official test subjects (2947 windows) are reused.
- Stages train:101 → eval:101 → train:102 → eval:102 → aggregate, each with a lock file, a hash-based skip rule and a pre-start budget guard (caps 1800 s wall / 3600 s process CPU, 2 threads). No stop condition fired (finite losses/logits; split and config hashes matched; no test window read before the checkpoint was frozen; resampled-arm input identical to C0, difference 0; identity checks ≤ 6.6e-5; 0 windows excluded).
- Results (subject means first, equal weight over 9; separate rows; never pooled; the original primary is not retro-updated):

  | run | step / calib. CE | CE C0 / acc. C0 | D = CE(mean,H8) − CE(time,H8) [subject bootstrap 95%] | d_s > 0 | ΔCE S8 / H8 mean / H8 time | S8 centered change: all / common-time / extra | inner | H8 norms ‖Δ‖ / ‖W‖ / ‖M‖ / ‖M_end‖ / ‖Q‖ |
  |---|---|---|---|---|---|---|---|---|
  | seed 0 (P1-HAR-01, unchanged) | 1200 / 0.092 | 0.270 / 0.904 | +0.016 [−0.004, +0.032] | 8/9 | −0.010 / +0.012 / −0.004 | 0.0290 / 0.0018 / 0.0284 | +0.091 | 7.69 / 7.44 / 0.997 / 0.013 / 0.565 |
  | seed 101 | 1600 / 0.120 | 0.252 / 0.916 | +0.050 [+0.023, +0.082] | 8/9 | +0.005 / +0.052 / +0.002 | 0.0231 / 0.0027 / 0.0216 | +0.215 | 9.99 / 9.60 / 0.712 / 0.031 / 0.457 |
  | seed 102 | 1600 / 0.083 | 0.252 / 0.923 | +0.044 [+0.011, +0.077] | 7/9 | −0.026 / +0.028 / −0.016 | 0.0445 / 0.0034 / 0.0451 | −0.142 | 5.92 / 5.77 / 1.366 / 0.016 / 0.771 |

  z_end flips no prediction in any run; S8 flips vs C0: 0.008 / 0.003 / 0.006. Reading: the separation (common-time part 8- to 16-fold below the extra-sampling part; W dominating the H8 token-mean change) is the same in all three initializations; the pooled-readout risk direction is the same, its size varies about three-fold; the two new per-run intervals exclude zero but share subjects and data with the original run, so PRIMARY_HAR_RISK_DIFFERENCE_NOT_ESTABLISHED (original pre-registered primary) is kept and no pooled statement is made. Not an independent-dataset replication. ARCHITECTURE_CLAIM_STOP kept.
- Cost ledger (`results/raw/R8-P1-HAR-REPLICATION/cost_ledger.json`): train:101 279.7 s wall / 522.0 s CPU; eval:101 81.1 / 149.5; train:102 232.6 / 459.9; eval:102 79.6 / 148.9; aggregate 1.1 / 0.9; process total 674.2 s wall / 1281.3 s CPU (caps 1800 / 3600; projection 550 / 1050). train:101 ran at 280 s against its 300 s allowance because bookkeeping commands (commit/push) ran concurrently on the same 2 threads; recorded, not re-run. PDF/package builds are outside this budget.
- Not done (not approved / not queued): a third seed, any protocol change, a per-token or cancellation diagnostic, pooling of runs, another model or dataset.

**Manuscript v5.2 (PDF built with `-halt-on-error -no-shell-escape`; see `results/pdf_check.json`).**

- Added: abstract sentence 5 (two further seeds, unpooled sensitivity check), contribution 3, §5.2 paragraph "Initialization sensitivity (two further model seeds)", §7 real-sensor scope (one architecture, three seeds on the same nine subjects), §8 clause, App. D paragraph and Tables (R8 risk, R8 decompositions; three runs as separate rows).
- Corrected: the App. D common-time sentence now says the common-time value is the change of the pooled readout over the 128 end points, not a per-token change, and that the per-token change was not computed; the "cancellation unlikely" speculation was removed (STATUS session 7 keeps the historical wording).
- Layout: the first build with the two R8 tables in §5.2 failed (13-column spec for a 14-column table) and, once fixed, ended the body on page 9. Fixed without any font/margin change: the two R8 tables moved to App. D, the §5.2 paragraph compressed, duplicated statements removed from §7 (margin convention and float32 caveat remain in §5.1), the P1-COMP-01 "9–20 % of the bound" sentence moved from Remark 4.3 to App. F (proof of Prop. 4.2), the head-of-mean identity checks moved to App. D "Checks", small trims in §1–§6. Body end (last Conclusion sentence) on page 8; 0 overfull boxes; 0 undefined references; abstract 5 sentences, 231 words by a plain count (math as one token). Visual check: all 19 pages of the final build rendered once with pdftoppm at 96 dpi and each page viewed (pages 1–19); no overlapping text, clipped table or broken float was seen; pages 11, 14, 17 and 19 end with white space because the following full-width tables moved to the next page (appendix; no page limit). PDF sha256 is recorded in `deliverables/P1_delivery.json`.
- Status words unchanged: ARCHITECTURE_CLAIM_STOP; PRIMARY_HAR_RISK_DIFFERENCE_NOT_ESTABLISHED; EXPLORATORY_FIXED_CHECKPOINT for P1-HAR-CT-01; SUBMISSION_READY=false; TARGET_YEAR=2027 on the unmodified 2026 kit.

**Review round (session 8): one read-only reviewer, 10-minute cap, fixed commit 6a3dfe1, one fix pass.** Numbers: every R8 value in main.tex, the two generated tables and the 34 Rviii macros was recomputed by the reviewer from `seed101/eval.jsonl`, `seed102/eval.jsonl`, P1-HAR-01 and P1-HAR-CT-01 raw files and the cost ledger with the project's subject-first rule and bootstrap: 0 mismatches; identity residuals re-derived (≤ 4.1e-5); no sentence pools the runs or updates the original primary; captions consistent with `summarize_run`. Blocking findings: 0. Non-blocking: 5. VALID_FIXED: (1) claims C42 section cited "Tables 6-7" (the R8 tables are Tables 13–14 in App. D); (2) Conclusion said the common-time part is "an order of magnitude smaller … in all three model seeds" although seed 101 gives 8.1-fold (seed 0 15.7, seed 102 13.2) → now "8- to 16-fold smaller"; (3) claims K4 section "App. C" → App. D; (4) the run identifier "R8-P1-HAR-REPLICATION" appeared in App. D and the table captions without the qualifier → "a sensitivity check on the same subjects and data, not an independent replication" added to the App. D paragraph and the Table 13 caption; (5) the PDF was rebuilt from the final source after the fix pass (this record's hashes refer to that build). NOTED: the moved P1-COMP-01 sentence sits at the end of App. F (proof of Prop. 4.2), which is where it belongs; the reviewer did not re-verify the 9–20 % figure (outside the R8 scope, time cap), so that figure keeps its session-5 verification status. Not presented as verification quality: the reviewer count.

**Packages and deliverables.** `scripts/make_export_bundle.py` and `scripts/make_anon_package.py` now include `results/raw/R8-P1-HAR-REPLICATION/` (both 116 KB checkpoints included, like the seed-0 HAR checkpoint); READMEs updated to the actual contents. `deliverables/P1_v5.2_<commit7>.pdf`, `P1_source.zip`, `P1_delivery.json` are written by `scripts/make_deliverables.py` after the final commit (DELIVERY_PENDING until filled in below).

**Next research decision (recorded, not queued).** The next informative step is one further architecture (a selective SSM with a content-dependent step, e.g. a Mamba-style block under the same protocol) or one naturally irregular dataset, because all HAR evidence so far is three initializations of one architecture on the same nine subjects; this is a decision for the user, and no experiment is queued.

## Session 7 (round 7): what was done

**Checkout.** HEAD at start 0e5cf3d (matched the report); clean tree; torch 2.14.0+cpu, pinned TIDES, TeX and the local HAR data/derived files still present. Hashes verified before use: `configs/p1_har_01.json` e603cf75…, `checkpoint.pt` e3b414f5…, runner 8d06bbd8….

**Decision kept.** No second seed, no second dataset, no analysis aimed at the 8/9 sign pattern. PRIMARY_HAR_RISK_DIFFERENCE_NOT_ESTABLISHED (D = +0.016 nats, interval includes 0) unchanged.

**P1-HAR-CT-01 (EXPLORATORY_FIXED_CHECKPOINT; post-hoc design, config and script committed before execution, f0f5506).**

- Question: of the pooled S8 change of the HAR checkpoint, how much is a change of the outputs at the 128 common physical time points vs the readout's averaging over the 896 added points.
- Stored raw first: per-token logits exist (C0 float16 in git; S8 float16 local). Their re-aggregation reproduces the stored float64 pooled logits only to 7.0e-3 / 3.3e-3 (float16), so one float32 streaming forward of C0 and S8 was run (batches of 256, eval mode, no intermediate tensors stored). It reproduces the stored C0 and S8 pooled logits exactly (0.0).
- Endpoint map: S8 token 8k (1-based) ↔ C0 token k, verified against time stamps and base-interval indices.
- Decomposition per window (exact, residual 0): z_all − z0 = (z_end − z0) + (z_all − z_end), with the official affine head; class centering Π applied to all terms alike.
- Results (subject means first, equal weight over 9; no interval, no test):

  | quantity | value |
  |---|---|
  | ‖Π(z_all − z0)‖ / ‖Π z0‖ (pooled S8 change) | 0.0291 |
  | common-time part ‖Π(z_end − z0)‖ / ‖Π z0‖ | **0.0018** (subjects 0.0010–0.0036) |
  | extra-sampling part ‖Π(z_all − z_end)‖ / ‖Π z0‖ | 0.0284 (subjects 0.0217–0.0324) |
  | inner product of the two centered terms | +0.091 logit² (subjects −0.156 to +0.490) |
  | final-feature change at the common end points | 0.0025 relative |
  | CE z0 / z_end / z_all | 0.2704 / 0.2705 / 0.2607 |
  | flips vs z0: z_end / z_all | 0.000 / 0.008 |

- fp64 check: pre-specified trigger (common-time part < 1e-3 or residual > 1e-4) not met → NOT_NEEDED; the 18 hash-selected window ids are recorded.
- Cost: 73.5 s wall / 115.2 s process CPU, 2 threads (timer before torch import and hashing); caps 1200 / 2400. Not measured: PDF/package builds, which are outside this budget.
- Interpretation limits: the extra-sampling term involves the model path through the added tokens (not a pure quadrature error); the common-time term is not attributed to a layer; norms are not shares (the inner product is reported).

**Review round (session 7): two read-only reviewers (methods; manuscript + packages), one round, one fix pass.** Numbers: all recomputed and matching. VALID_FIXED: share-like "of which / the rest" wording in the abstract, conclusion and K4 (terms are not additive shares; no quadrature attribution); the relative-norm definition (means of per-window ratios, subject-first; ratio of subject-mean norms 0.093/30.4 = 0.0031 added); the identity residual described as zero by construction; the fp64 rule stated with its threshold and the absolute comparison (0.093 logits ≈ 3000× the float32 pooling residual); C41 0.0291 → 0.0290; the HAR diagnostic added to the post-hoc list; the per-token common-time change recorded as not computed (per-end-point feature change 0.0025 is of the same order as the pooled 0.0018, so cancellation is unlikely); the anonymous package's hard-coded "v4" header label. NOTED, not changed (would alter an executed config or raw): the config/raw phrase "about 1e-3" for the float16 gap (observed 7.0e-3 / 3.3e-3); the fp64 rule's threshold is relative while its motivation was absolute (decision unaffected); `git_dirty=true` at run time comes from the run's own untracked outputs; the ledger's thread field is a literal 2 and torch threads are set after Stage A (numpy only; cpu/wall 1.57); the config was committed 5 s before the run in the same session.

**Manuscript v5.1 (PDF built; see pdf_check.json).**

- §5.2: the pooled S8 change 0.029 is no longer called a coupling effect; it is linked to the new common-time result (0.0018). The affine-head identity is written as A(Σ w h) = Σ w A(h), Σ w = 1.
- Abstract: one paragraph, 5 sentences, 223 words (tex count, math as one token), three numbers, non-significance kept; no "96.7 % explained" share.
- §7: synthetic-evidence scope separated from the HAR scope. App. A: "no external data" limited to the synthetic stages. §G: HAR checkpoint included in the packages, synthetic checkpoints excluded (hashes + retrain command) — verified against the actual anonymous package contents.
- Forbidden moves not made: no sign test as primary, no subject exclusion, no added seeds.

## Session 6 (round 6): what was done

**Checkout.** Branch `claude/intelligent-maxwell-t1lcda`; HEAD at start dc426eb (matched the report); clean tree.

**Authorization (delivered directly by the user in the round-6 instruction).**

- Approved and used: one download of the official UCI HAR ZIP plus page metadata/license; the existing torch-CPU / pinned-TIDES / TeX environment; one TIDES classifier with one model seed and one linear control; 2 CPU threads; 3600 s wall / 7200 s process-CPU total for smoke + train/dev/test + bootstrap.
- Not approved and not used: GPU, paid APIs, new weights, extra libraries, other datasets, extra synthetic seeds.
- `/nvmedata` does not exist in this sandbox; the checkout was used; `/data000` untouched.

**A. Manuscript corrections without training.**

1. `tab_newobs` caption: "genuine new external information" → "change in the held-input reconstruction; no task or Bayesian information gain is implied".
2. Eq. (2): $T_m = D_m - D_1$ defined; norm ratios never read as shares or causal contributions.
3. Intervals stated as pointwise per estimand and seed, unadjusted for secondary conditions; a non-negative discrepancy's interval excluding zero is nonzero-ness, not a cause.
4. Prop. 4.2 unchanged (scalar LTI upper bound); its proof moved to an appendix for space.
5. Anonymization check of the existing packages: the TIDES pin (4b51adc…) and every checkpoint sha256 are kept; only this repository's commit ids were replaced. No change needed.
6. Layout: Reproducibility moved before the toy tables; pointer sentence under the toy-tables heading.

**B. P1-HAR-01 (UCI HAR), one run.**

- **Data.** Official ZIP, 61,005,872 bytes, sha256 c00b8030…; nested archive checked for path traversal before extraction; license: page says CC BY 4.0, README requires citation and prohibits commercial use (both recorded); data kept in gitignored `data/`. Nine inertial channels; the 561-feature tables are not used. 7352 train / 2947 test windows as distributed.
- **Contract.** Official subject partition kept; calibration subjects 7, 15, 17, 19 by an ID-hash rule (re-derived by the runner); fit subjects 17 (5988 windows); normalization fit-only; 128 held intervals of 1/50 s (T = 2.56 s); C0 128 / S8 1024 / H8 576 tokens; no mask or padding; the resampler reads only observed values; the subject is the analysis unit.
- **Model.** Official `TIDESClassifier` with the pilot backbone setting (23,786 parameters), 9 inputs, 6 classes; the classifier's defaults for `conj_sym`, `encoder_depth`, `proj_norm` overridden to reuse the existing setting (documented).
- **Pre-registration.** Config and runner committed before execution (97963e4); an independent 4-lens review found no leakage; its findings were applied as one pre-execution amendment (6a8ab2f, before training): projection-aware budget guards, same-path arms (head of pooled features), test-opened lock, enforced eval-mode checks, finiteness guards, per-subject exclusion counts, degenerate-interval verdict logic.
- **Budget ledger (all stages, 2 threads):**

  | stage | wall s | CPU s | note |
  |---|---|---|---|
  | data | 2.0 | 2.2 | parse + hash |
  | smoke (first code) | 16.0 | 14.6 | 20 updates, model discarded |
  | smoke (amended) | 13.4 | 24.1 | decision: 2000 updates OK |
  | train | 183.0 | 354.2 | 2000 updates; selected update 1200 (cal CE 0.092, acc 0.974) |
  | eval | 169.6 | 317.7 | C0/S8/H8 × 3 arms + timing |
  | control | 3.7 | 4.6 | |
  | aggregate (×2) | 0.8 | 1.2 | second run only adds provenance |
  | **total** | **388.4** | **718.6** | cap 3600 / 7200; stage timers of the code that ran started after imports/hashing (≈2–4 s per stage unrecorded; true total ≈ 400 s wall) |

  Outside this budget: download 33 s, PDF and package builds.
- **Checks.** Resampled input = C0 exactly on S8/H8 and its logits bit-identical to C0 (0.0); head-of-mean = direct forward (0.0); decomposition identities to 2.7e-5 (float32); no non-finite value; 0 windows excluded by the centered-logit rule.
- **Results (9 test subjects, subject means first):**
  - Primary D = CE(token mean, H8) − CE(time-weighted, H8) = **+0.016 nats, 9-subject bootstrap [−0.004, +0.032], 8 of 9 subjects positive → interval includes 0: NOT ESTABLISHED** (and not evidence of equivalence).
  - Model coupling (S8): class-centered logit change 0.029 (every subject 0.022–0.033), TV 0.008, 0.8 % prediction flips; ΔCE −0.010 [−0.034, +0.012].
  - Pooled weighting (H8): token mean: centered-logit 0.210, flips 2.8 %, ΔCE +0.012 [−0.005, +0.028]; time-weighted: 0.016, 0.3 %, −0.004 [−0.017, +0.007]. Decomposition norms: ‖p_mean − p_C0‖ 7.69, ‖W‖ 7.44, ‖M‖ 1.00, ‖M_end‖ 0.013, ‖Q‖ 0.57 → again mostly the weighting identity.
  - Accuracy 0.904 (C0), 0.905 (H8 mean), 0.903 (H8 time).
  - Control: features identical to 1.8e-14; CE 0.695 / acc 0.765, unchanged across grids.
  - Cost: native_S8_e2e / resampled_S8_e2e = 12.7 (forward-only 12.4); H8 5.0.
- **Stored.** `results/raw/P1-HAR-01/`: eval.jsonl (9.8 MB, per-window float64 pooled logits, CE, predictions, distances, decomposition), C0 token logits float16 gz (4.1 MB), checkpoint (116 KB) + sha256, train/smoke/control/aggregate/ledger/acquisition. S8/H8 token logits local only. Note: the config's "about 30 KB" checkpoint size estimate was wrong (116 KB); the config was not edited after training because its sha256 is verified by every later stage.

**C. Manuscript v5.** §5 split into 5.1 (synthetic repetition, compressed; layer diagnostic and one-fix table moved to appendices) and 5.2 (HAR); abstract, contributions, conclusion, limitations, Impact Statement and a new appendix (data, protocol, cost, timing) updated; dataset citation added through the bib provenance (manual entry, source recorded). claims.csv: K4 added; K1/K2 notes extended. PDF: 17 pages, body's last sentence on page 8, 0 overfull boxes, 0 undefined references, anonymity check clean; pages 4–8 rendered and inspected at 96 dpi.

**D. Review workflows.** Four independent pre-execution reviewers (leakage, representation, statistics, engineering), plus a post-build claim verification.

- Leakage and representation lenses reported before training; no leakage found; their findings became the pre-execution amendment (6a8ab2f).
- The statistics lens reported during evaluation; its findings were applied to the aggregate/control stages before those ran (4be0c87).
- The engineering lens reported after the run. Its findings concern robustness, not results: crash paths left no ledger row; `started_utc` was the finish time and stage timers started after imports/hashing; eval/control/aggregate outputs lacked code provenance; a NaN in the smoke FAIL path; no guard on the data stage. All were fixed in the runner afterwards (133c434); aggregate was re-run once only to add provenance (every statistic byte-identical; the extra ledger row is kept).
- Code provenance of the run: data/smoke/train used the runner at 60536b3 (train.json's `git_head` fa3c64c is the HEAD at the end of training; the commit in between touched only the asset generator and manuscript); eval ran with the runner at 60536b3; control and the first aggregate with 4be0c87.

## Session 5 (round 5): what was done

**Checkout.** Branch `claude/intelligent-maxwell-t1lcda`; HEAD at start 2fd9ae6, matching the previous report.

**Authorization.**

- The round-5 instruction is not an approval for downloads, GPU, paid APIs, global changes, external upload or submission, and no separate resource-approval block was delivered.
- Only the existing CPU environment was reused: torch 2.14.0+cpu, the pinned TIDES clone and TeX Live.
- The only new computation is the frozen forward pass that section B of the instruction prescribes when per-token outputs are missing.
- `/nvmedata` does not exist in this sandbox, so nothing was created there; `/data000` was not touched.

**A. Corrections to v3 (no retraining).**

1. **Body end page.** v3 ended its body on page 7, not page 6. `check_pdf.py` now reads the `body-end` label, which sits after the last Conclusion sentence, from `main.aux` and cross-checks the last words in the PDF text. v4: body ends on page 7 of 16.
2. **Table readability.**
   - Every `\resizebox` was removed from the generated tables; `check_paper.py` now fails if a table uses `\resizebox`, `\scalebox`, `\tiny` or `\scriptsize`.
   - A probe build had shown effective font sizes of 5.2–6.6 pt in the v3 body tables (Table 2 at 6.3 pt).
   - The body now has three tables at 9 pt (`\small`), none scaled:
     - model coupling;
     - readout decomposition;
     - task loss.
   - Timing is in the appendix, the development row is in the appendix (`tab_dev_newtest`), and the toy tables `tab_split`, `tab_coupling` and `tab_layers` moved to the appendix.
   - The figure fonts were enlarged (ticks and labels 8 pt, legend 7 pt; the legend was 5 pt).
   - Pages 1, 5, 6, 7 and 9–13 were rendered and inspected at 96 dpi.
3. **"only ... together".** Replaced by "in the tested single-layer ablation, the combined corrections reduced …", with an explicit statement that this is not a necessity result: a last-state readout needs no time integral, and a zero input removes the artifact.
4. **Prop. 4.2.** Stated as an O(mesh) upper bound for the scalar LTI cascade only. It gives no lower bound (zero input gives zero error) and no deep-network theorem. "The pattern … predicts" became "qualitatively consistent with". Also corrected: conditions (ii)–(iv) can hold in every layer, but condition (i) holds only in layer 1.
5. **§4.2.** Invariance defect and approximation error are now separated. A fixed grid removes the defect but not the approximation error.
6. **"New information".** Restricted to changed observed values and input reconstruction. An artifact is defined operationally as a violation of the pre-declared same-held-path invariance.
7. **Precision.** The 1e-4 threshold from the scalar 16,384-sub-step emulation is no longer used as a floor for the trained network. The checkpoint-specific fp32/fp64 comparison from the stored double-precision values is reported instead:
   - mean-estimate difference ≤ 5.2e-6 (relative 3.8e-4);
   - per-trajectory difference ≤ 1.8e-5;
   - smallest per-trajectory discrepancy 1.6e-3.
8. **"Removes".** "Time-weighted pooling removes the readout effect" was withdrawn. The paper now reports the reduction (mean |change| 0.60–0.62 → 0.013–0.023) and the residual. The "exact integral" is described as the time average computed in closed form in this implementation, not as a quantity defined only for exact ZOH.
9. **Unproved sketches and reproducibility section.** Appendix F was removed and moved to `notes/unproved_sketches.md`. The reproducibility section now matches the packages: no checkpoints, no third-party code, checkpoint sha256 listed.

**B. H8 signed decomposition** (`configs/p1_real_02_h8_decomp.json`, fixed at 2c4b865 before execution):

- **Outputs not preserved.** P1-REAL-02 had not stored per-token outputs, so one frozen native forward per existing checkpoint was run on C0 and H8. This covered 5 checkpoints: training seeds 1–4 and the development checkpoint.
- **Cost:** 7.7 s wall / 7.7 s CPU in the script, 13.5 s in the shell, 2 threads. There was no training and no timing.
- **Validity check:** stored p_mean, p_time and p_C0 were reproduced exactly (maximum difference 0). The identity residual is ≤ 4.6e-15.
- **Result (seeds 1–4):**

  | Quantity | Value |
  |---|---|
  | mean signed change p_mean(H8) − p(C0) | +0.46 … +0.50 |
  | density-weighting term W = (7/18)(A − B), from the C0 outputs alone | +0.48 … +0.49 |
  | sign agreement of W with the change | ≥ 98 % of trajectories |
  | model term M | −0.016 … +0.011 |
  | \|M\| | 0.023–0.028 |
  | \|M\| on interval-end tokens | 0.0005–0.0049 |
  | time-weighted \|Q\| | 0.013–0.023 |

- **Interpretation.** The primary D_s effect is therefore mostly the weighted-average identity of token-mean pooling applied to an output whose first-half mean is larger (A − B = 1.22–1.26). It is not a selective-SSM-specific failure.
- **Limits.** The signed terms are not an additive decomposition of D_s.
- **Outputs:**
  - `results/raw/P1-REAL-02/h8_decomposition/`;
  - per-token outputs stored as gzip JSON, so later analyses need no forward pass.

**C. P1-HAR-01 (UCI HAR, DOI 10.24432/C54S4K): BLOCKED / NOT_RUN.**

- No resource-approval block was delivered for this round.
- Nothing was downloaded; the dataset page, zip, README and license were not accessed.
- No config was written, because the round-5 instruction says to stop producing documents automatically.
- Running it needs explicit approval for:
  - downloading the official zip and recording its hash;
  - CPU training of one TIDES classifier with one seed.

**D. Manuscript v4.**

- The title was kept.
- The abstract and conclusion now separate three effects: model coupling, pooled weighting and task risk.
- The scope is stated as one selective SSM (TIDES) on one synthetic task.
- `claims.csv`: 13 rows updated (K1, K2, K3, C14, C17, C18, C19–C22, C27, C37, C38); no other row changed.
- PDF: 16 pages. The body's last sentence is on page 7, references start on page 7, the appendix starts on page 9. There are 0 overfull boxes and 0 undefined references, and the anonymity check of the PDF is clean.
- ICML 2027: the conference page, its CallForPapers page and the `icml2027.zip` style URL all returned 404 on 2026-09-27. The unmodified ICML 2026 style is therefore kept, with TARGET_YEAR=2027, TEMPLATE_YEAR=2026 and SUBMISSION_READY=false.

**Packages.**

- **`export_bundle/`** is the internal evidence package, built by `scripts/make_export_bundle.py`.
- **`anon_submission/`** is the anonymous review package, built by `scripts/make_anon_package.py`, with its check in `results/anon_check.json`.
  - Repository commit ids were replaced by `<commit>`.
  - The internal checker and packaging scripts, the internal documents and the claims ledger are excluded.
  - Checks passed:
    - no identifying pattern, with patterns derived from the git remote, the authors, the commit hashes and the local paths;
    - every README "Included" path exists;
    - the generator re-run on a copy of the package reproduces `paper/generated/` byte for byte.
  - Checkpoints and third-party code are not included in either package; the reproduction commands and sha256 are given instead.

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
- ~~The main body ends on page 6 (Conclusion).~~ **CORRECTED in session 5:** page 6 was where the Conclusion heading appeared, but the Conclusion text ran onto page 7. The v3 body ended on page 7, which is within the 8-page limit. `scripts/check_pdf.py` now counts the page of the last body sentence.
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

**Session 7:** no further run is proposed by this session. The owner's open choice remains the one of session 6 (a second seed or dataset under the frozen P1-HAR-01 protocol, or stopping at the current scope); the common-time diagnostic does not change it.

*Session 6 record:* second seed / second dataset decision; nothing assumed approved.

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
  - 11 pages in total; the main body was reported to end on page 6, within the 8-page limit. This was counted by heading, not by last sentence, and was not re-verified in session 5.
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

