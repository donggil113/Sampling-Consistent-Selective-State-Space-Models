# Related work: time-grid consistency of (selective) SSMs

First check: 2026-09-26 (session 1). Updated: 2026-09-26 (session 2).

Access levels:

- `FULL_TEXT_SECTIONS`: the listed sections of the arXiv HTML full text were read.
- `KEYWORD_CHECK`: the full text was searched and the matching passages were read.
- `ABSTRACT_ONLY`.
- `NOT_ACCESSED`.

Provenance:

- `self`: checked directly by the main author.
- `agent`: read only by the literature sub-agent of session 1. Re-verify an agent-only entry before relying on it.

Bibliographic metadata (title, authors, year) comes from arXiv abs-page `citation_*` tags, stored in `paper/bib_provenance.json`. No venue is asserted in the bibliography except the arXiv journal-ref field, where present.

## What is already known (must NOT be claimed as a contribution)

| Known result | Where (provenance) |
|---|---|
| Physical Δt in the discretization; irregular sampling via per-step Δt | LSSL (self: "irregularly-spaced data can be handled by discretizing the same matrix A using a different timescale Δt"); S5 §6.3 (self); TIDES (self) |
| Test-time Δ rescaling for a new sampling rate or resolution | S4 §4.3 (self); S5 §6.2, Table 2, 16k→8k by naive decimation (self); S4ND (self: "only the relative rate matters"); Zubić et al. §4.3.3 (self: heading) |
| ZOH is exact for held inputs; flow composition | textbook; S5 Eq. 6 (self); TIDES App. B.1 (self); Mamba-3 §3.1.1 (self) |
| Released Mamba-1/2 code uses exponential-Euler, B̄=ΔB | Mamba reference code `deltaB_u = delta*B*u` (self); Mamba-3 §3.1.1 Eq. 4 (self) |
| Exponential-trapezoidal rule, second order only if λ_t = ½ + O(Δ_t) | Mamba-3 Prop. 1 and Remark 3 (self) |
| Mamba's Δ is a content gate; a Mamba surrogate with Δ as an input channel fails outside the training Δ range | TIDES §2.3, §4 (self) |
| Physical Δ + exact ZOH + selectivity on Re(Λ), B, C | TIDES §3.1, Fig. 3 (self) |
| A continuous-time reading of selective SSMs needs Δ ∝ dt | Cirone et al. (self: "δ plays the role of the differential dt") |
| Neural CDEs driven by an interpolated path; robust to irregular sampling | Kidger et al. App. A.3, natural cubic splines (self); Log-NCDE intro (self) |

## Per-paper record

| Paper | arXiv | Access | Provenance | Relevant facts | Known vs. gap |
|---|---|---|---|---|---|
| TIDES (Soydan, Bessa, Mohr, Barreira; 2026 preprint) | 2605.09742 | FULL_TEXT_SECTIONS: abstract, §1–3.2, §4, §5 intro, §5.3, §6–8, App. A, B, C | **self** (session 2) + code | See the list below | **closest prior work**; architecture is prior art |
| Mamba-3 (Lahoti et al., 2026) | 2603.15569 | FULL_TEXT_SECTIONS: §3 intro, §3.1–3.1.2 (Table 1, Eq. 4–6, Prop. 1, Remarks 2–4); section headings of §4; KEYWORD_CHECK for "irregular", "sampling rate", "resolution", "time series": 0 hits | **self** | Right-endpoint ZOH derivation for LTV systems; exponential-Euler = released Mamba-1/2 rule; exponential-trapezoidal with data-dependent λ_t; §4 covers language modeling, retrieval, ablations, inference efficiency and kernels. Official module `mamba_ssm/modules/mamba3.py` imports Triton kernels (self) | closest on the discretization order; no grid tests found |
| S5 (Smith, Warrington, Linderman) | 2208.04933 | FULL_TEXT_SECTIONS: §3.2 (Eq. 6), §6.2, §6.3, App. B.1.3, App. F.3 | **self** + code | ZOH; global Δ rescaling for 8 kHz; per-step Δ_t for the irregular pendulum; ablations S5-drop and S5-append; timescale initialization [0.001, 0.1]. Official code on the main branch uses a static learned step; the pendulum code is on a separate branch; `requirements_cpu.txt` pins jax 0.3.5 (self) | strongest direct LTI baseline |
| S4 (Gu, Goel, Ré) | 2111.00396 | KEYWORD_CHECK §4.3 "Sampling resolution change" paragraph | self (agent: more sections) | Bilinear discretization; 96.3% on SC10 at 0.5× frequency without re-training, by changing Δ | known |
| LSSL (Gu et al.) | 2110.13985 | KEYWORD_CHECK | self (agent: §2, §3.2, §5.2–5.3) | Irregular data via a different Δt | known |
| S4D (Gu et al.) | 2206.11893 | KEYWORD_CHECK (bilinear/ZOH equation) | self (agent: §3.1, §5.1) | Both bilinear and ZOH rules | known |
| Mamba (Gu & Dao) | 2312.00752 | KEYWORD_CHECK ("zero-order hold") + code | self | Paper states the ZOH rule; code uses B̄=ΔB; `d_conv=4` conv1d; dt bias initialized so that softplus(bias) ∈ [0.001, 0.1] | known |
| Mamba-2 / SSD (Dao & Gu) | 2405.21060 | KEYWORD_CHECK | self (agent: §2, §7.1, §8.1) | "adopt the same parameterization and discretization step as prior work" | known |
| Liquid-S4 (Hasani et al.) | 2209.12951 | KEYWORD_CHECK | self (agent: Eqs. 8–9, §4.2) | Discusses the zero-shot 8 kHz test setting | known |
| Cirone, Orvieto, Walker, Salvi, Lyons | 2402.19047 | KEYWORD_CHECK | self (agent: §2.1, §3.1, §5–7, App. C.1.1, F) | δ plays the role of dt (Euler limit) | theory known |
| Neural CDE (Kidger et al.) | 2005.08926 | KEYWORD_CHECK App. A.3 | self (agent: §3, §6.1, §6.3) | Natural cubic spline path | known |
| Log-NCDE (Walker et al.) | 2402.18512 | KEYWORD_CHECK intro | self | NCDEs are robust to irregular sampling and decouple network evaluations from observations; arXiv journal-ref present | known |
| S4ND (Nguyen et al.) | 2210.06583 | KEYWORD_CHECK | self (agent: §1, §3, §4.2) | "only the relative rate matters" | known |
| Event-camera SSMs (Zubić et al.) | 2402.15584 | KEYWORD_CHECK (§4.3.3 heading) | self (agent: §1, §3.3, §4.3.1) | Evaluation at different frequencies; arXiv journal-ref present | known |
| LinOSS (Rusch & Rus) | 2410.03943 | KEYWORD_CHECK (headings "Implicit time integration", "IMEX") | self (agent: §2.3, §3) | Implicit and IMEX integration | tangential |
| S7 (Soydan et al., 2024) | 2410.03464 | FULL_TEXT_SECTIONS §2–3.4 | **agent only** | Asynchronous discretization (heading verified by self) | not cited in the manuscript |
| Δt-Mamba3D (Zhou et al., 2025) | 2510.19003 | FULL_TEXT_SECTIONS (method) | **agent only** | Affine Δt scaling formula (not verified by self) | not cited in the manuscript |
| SeRpEnt (Rando et al., 2025) | 2501.11729 | FULL_TEXT_SECTIONS §1–3 | agent only | — | not cited |
| SS-NO (Bendahi et al., 2026) | 2605.18905 | ABSTRACT_ONLY + §4 | agent only | — | not cited |
| MambaNO | — | NOT_ACCESSED | — | — | not cited |

### TIDES: facts checked directly (session 2)

- **Paper.**
  - Eq. 2 and Eq. 9 are written left-endpoint: Δ_k = t_{k+1} − t_k drives u_k.
  - Eq. 4 is the ZOH recurrence with input-dependent Λ_k. Eq. 5 gives the selectivity heads.
  - Fig. 3: B̄_k = Λ_k⁻¹(Λ̄_k − I)B_k.
  - App. A: per-mode learnable timescale initialized in [0.001, 0.1]; BatchNorm(affine=False) over batch and time.
  - Fading Flash (§4, App. C) uses L=40 with a global Δ; training Δ ∈ [0.5, 1.5]; test Δ ∈ {0.1, …, 2.0}.
  - Random drop (§5.3) is run on EigenWorms and reports accuracy. Compute: about 5,000 A100 GPU-hours.
  - A full-text keyword search found 0 hits for "refine", "upsampl", "sub-step" and "splitting".
- **Code** @ 4b51adc, MIT.
  - The step is `step_scale * exp(log_step)`.
  - The scan uses the gap *before* step k, i.e. the right-endpoint convention.
  - B̄ is computed as `(1/Lambda)*(Lambda_bar-1)*B`.
  - An optional causal conv is off by default.
  - The classifier uses `h.mean(dim=1)`.
  - No checkpoints are released.

## Remaining difference relative to this project (stated cautiously)

In the sources above we found no report of the following. This is absence of evidence, not proof of novelty.

1. A controlled refinement of the *same* held path for selective SSMs that separates splitting from new observations and decomposes the change exactly, with explicit denominators and the cross term.
2. Readouts (sample-count vs. time-weighted vs. exact integral) and inter-layer resampling treated as entry points of grid dependence.
3. A measurement of these entry points in a trained official implementation. This is P1-REAL-01, NOT_RUN.

Consequence: the contribution is framed as *measurement and decomposition* only. The new-architecture claim is STOPPED.
