# Related work: time-grid consistency of (selective) SSMs

Checked 2026-09-26. Access levels:

- `FULL_TEXT_SECTIONS`: the listed sections of the arXiv HTML full text were read, located by keyword search. The whole paper was not read.
- `ABSTRACT_ONLY`: only the abstract was read.
- `NOT_ACCESSED`: the paper could not be read.

Provenance:

- `agent`: read by the literature sub-agent of this session. The main author did not re-read it.
- `self`: re-checked directly by the main author.

Anything tagged only `agent` should be re-verified before it is cited in a manuscript.

## What is already known (must NOT be claimed as a contribution)

| Known result | Where |
|---|---|
| Feeding the physical step Δt into the discretization; irregular sampling via per-step Δt | LSSL §2/§5.2; S5 §6.3 (irregular pendulum); S7 Eq. 12; TIDES |
| Changing Δ at test time for a new sampling rate / resolution | S4 §4.3; LSSL Table 2; S5 Table 2 (16k→8k by Δ rescaling); S4ND; Zubić et al. 2024 |
| ZOH is exact for held inputs; flow composition / semigroup law | textbook; S5 Eq. 6; Cirone et al. App. F; TIDES App. B.1 |
| Mamba's implementation uses Euler for B (B̄ = ΔB) while the paper states ZOH | Mamba reference code (`selective_scan_ref`: `deltaB_u = delta*B*u`) [self]; Mamba-3 §3.1 [agent]; Cirone et al. Eq. 3 [agent] |
| Mamba's Δ is a content gate, not physical time, and this limits irregular-sampling generalization | TIDES abstract [self], §4–5.3 [self: keyword search; agent: full read] |
| An architecture that keeps Δ physical, uses exact ZOH, and puts selectivity on the diagonal state matrix | TIDES (arXiv 2605.09742) [self: abstract + App. B.1 keyword check] |
| A continuous-time (linear CDE) reading of selective SSMs requires Δ ∝ dt | Cirone et al. §3.1 [agent] |
| Neural CDEs are invariant to time reparametrization; time and observation intensity are added as channels | Kidger et al. §3.5, §6.1 [agent] |
| Aliasing under resolution change, with bandlimiting as the fix | S4ND §4.2; Zubić et al. 2024 [agent] |
| Mamba-3 exponential-trapezoidal discretization, and the short conv made optional | Mamba-3 §3.1 [agent]; abstract [self] |

## Per-paper record

| Paper | arXiv | Access | Provenance | Relevant facts | Known vs. gap |
|---|---|---|---|---|---|
| S4 (Gu, Goel, Ré, ICLR 2022) | 2111.00396 | FULL_TEXT_SECTIONS (§2.3, §4.3, App. C) | agent | Bilinear discretization; SC10 at 0.5× frequency without retraining, obtained by changing Δ | known |
| LSSL (Gu et al., NeurIPS 2021) | 2110.13985 | FULL_TEXT_SECTIONS (§2, §3.2, §5.2, §5.3) | agent | Generalized bilinear transform; irregular data via a different Δt; test-time rate change | known |
| S4D (Gu et al.) | 2206.11893 | FULL_TEXT_SECTIONS (§3.1, §5.1) | agent | ZOH and bilinear formulas; discretization choice has no noticeable effect on the benchmarks | known |
| S5 (Smith et al., ICLR 2023) | 2208.04933 | FULL_TEXT_SECTIONS (§3, Eq. 6, §6.2–6.3, App. F.3) | agent | ZOH with Λ̄=e^{ΛΔ}, B̄=Λ⁻¹(Λ̄−I)B̃; per-step Δt for the irregular pendulum; ablations S5-drop and S5-append (Δt as an input feature) | known; S5 is the strongest direct LTI baseline |
| Mamba (Gu & Dao 2023) | 2312.00752 | FULL_TEXT_SECTIONS (§2, §3.2, §3.4–3.6) [agent] + code [self] | self (code) / agent (paper) | Code [self]: `deltaA = exp(delta*A)`, `deltaB_u = delta*B*u`, `d_conv=4` depthwise `nn.Conv1d`, Δ = softplus(dt_proj(·)+bias), softplus(bias) initialized in [0.001, 0.1]; no physical time | Euler-B known; no measurement of refinement effects found |
| Mamba-2 / SSD (Dao & Gu, ICML 2024) | 2405.21060 | FULL_TEXT_SECTIONS (§2, §7.1, §8.1) + code | agent | Same discretization as Mamba-1; depthwise conv1d with `d_conv=4` | known |
| Mamba-3 (Lahoti et al., ICLR 2026) | 2603.15569 | abstract [self]; FULL_TEXT_SECTIONS §1, §3.1 [agent] | self (abstract) / agent | [agent] Says Mamba-1/2 are "exponential-Euler"; proposes exponential-trapezoidal; short conv optional; LM evaluation only | closest work on the Euler-vs-ZOH-vs-trapezoid question; no grid tests (per agent) |
| TIDES (Soydan, Bessa, Mohr, Barreira, 2026 preprint) | 2605.09742 | abstract [self]; keyword search of full text [self]; FULL_TEXT_SECTIONS §1–8, App. A–C, F [agent] | self + agent | [self] Δ̃≡Δ kept physical; input dependence moved to the diagonal state matrix; ZOH exact-solution derivation (App. B.1); "Fading Flash" OOD-Δ benchmark; random-drop on EigenWorms. [self] Full-text keyword search found 0 hits for "refine", "upsampl", "sub-step", "splitting", "pooling" | **closest prior work.** It already occupies the "physical Δ + exact ZOH + selectivity elsewhere" architecture |
| Liquid-S4 (Hasani et al.) | 2209.12951 | FULL_TEXT_SECTIONS (Eqs. 8–9, §4.2) | agent | Input-dependent transition; weaker zero-shot 8 kHz transfer | early evidence that input dependence hurts rate transfer |
| Cirone, Orvieto, Walker, Salvi, Lyons (NeurIPS 2024) | 2402.19047 | FULL_TEXT_SECTIONS (§2.1, §3.1, §5–7, App. C.1.1, F) | agent | Selective SSMs as linear CDEs only in a Δ ∝ dt limit; ZOH exact for piecewise-linear drivers | theory known; nothing measured |
| Neural CDE (Kidger et al., NeurIPS 2020) | 2005.08926 | FULL_TEXT_SECTIONS (§3, §3.5–3.6, §4.2, §6.1, §6.3) | agent | Spline path (non-causal); speed-blind invariance; time added as a channel | known |
| Log-NCDE (Walker et al., ICML 2024) | 2402.18512 | FULL_TEXT_SECTIONS (§1, §2.5, §3.3) | agent | Solver steps decoupled from the number of observations | known |
| SLiCE (Walker et al. 2025) | 2505.17761 | FULL_TEXT_SECTIONS (§4.6, experiments, appendix) | agent | Structured linear CDE; first-order flow approximation | known |
| S7 (Soydan et al. 2024) | 2410.03464 | FULL_TEXT_SECTIONS (§2–3.4) | agent | Physical event Δt in the decay, but the input term is not scaled by Δt, so it is not refinement-consistent | related |
| S4ND (Nguyen et al., NeurIPS 2022) | 2210.06583 | FULL_TEXT_SECTIONS (§1, §3, §4.2) | agent | Resolution change via Δ; aliasing; LTI only | known |
| LinOSS (Rusch & Rus, ICLR 2025) | 2410.03943 | FULL_TEXT_SECTIONS (§2.3, §3) | agent | Implicit / IMEX discretization with a fixed Δt | not about irregular grids |
| Event-camera SSMs (Zubić et al., CVPR 2024) | 2402.15584 | FULL_TEXT_SECTIONS (§1, §3.3, §4.3.1) | agent | Inference-frequency change via Δ scaling plus bandlimiting; S4/S4D/S5 only | known |
| Δt-Mamba3D (Zhou et al. 2025) | 2510.19003 | FULL_TEXT_SECTIONS (method) | agent | δ·(1+γΔt/τ_min): affine in Δt, not proportional, so not refinement-consistent | related |
| SeRpEnt (Rando et al. 2025) | 2501.11729 | FULL_TEXT_SECTIONS (§1–3) | agent | Reads Δ as information-driven "time intervals" | tangential |
| SS-NO discretization error (Bendahi et al. 2026) | 2605.18905 | ABSTRACT_ONLY + §4 | agent | LTI convolution kernels only | tangential |
| MambaNO (NeurIPS 2024) | — | NOT_ACCESSED | agent | — | unknown |

## Remaining difference relative to this project (stated cautiously)

In the sources above we found no report of the following. This is absence of evidence, not proof of novelty.

1. A controlled refinement test of the *same* held path (1/2/4/8 ZOH sub-steps) for selective SSMs, with the change decomposed into Δ-scaling, B-discretization, sample-count vs time-weighted readout, token-indexed conv1d, and input reconstruction.
2. Separating *splitting* (path unchanged) from *new observations* (path changed), so that a genuine information change is not counted as an artifact.
3. The claim that, in stacked layers, splitting at the input turns into *new observations* at deeper layers. This is a CONJECTURE, not tested; see RESEARCH_PACKET.md §5.

Consequence: any contribution must be framed as *measurement and decomposition* plus the *simplest fix that suffices*. It must not be framed as a new architecture. TIDES already occupies the "physical Δ + exact ZOH + selectivity moved to Λ" design, and the FIRST_RUN toy shows that simple time handling removes the single-layer artifacts (STATUS.md).
