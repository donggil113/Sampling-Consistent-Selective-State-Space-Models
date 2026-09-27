# Internal research note: unproved sketches (removed from the manuscript in v4)

These items were Appendix F ("Proof sketches (unproved)") of manuscript v3 (git 1bca23e). They were removed from the paper in v4 because they are not proved and are not needed for any claim.

- **Status:** UNPROVED / CONJECTURE. None of them is a result of this project.
- **Scope of the paper:** the paper uses only Proposition 4.1 (standard, proved) and Proposition 4.2 (proved; an upper bound for a scalar LTI cascade).
- **Also kept in the paper:** the one-line local Euler-B inequality |phi_1(z) − 1| ≤ |z|/2 for Re z ≤ 0, which is proved inline in §4.1.

## Assumptions used by the sketches

- **(A1)** Re λ_n ≤ −μ < 0.
- **(A2)** g : [−U, U] → [g_min, g_max], with g_min > 0 and Lipschitz constant L_g.
- **(A3)** F(v) = g(v) B(v) v is L_F-Lipschitz, with |F| ≤ F_max.
- **(A4)** u is L_u-Lipschitz, with |u| ≤ U.

## Hold error (UNPROVED sketch)

**Setting.** For the solutions driven by the held input ũ and the continuous input u, the error e satisfies

  ė = g(ũ) λ e + [ (g(ũ) − g(u)) λ h_u + F(ũ) − F(u) ].

**Bounds used.**

- The bracket is at most (L_g |λ| H + L_F) L_u δ, where H = F_max / (g_min μ).
- The propagator is bounded by exp(−g_min μ (t − s)).

**Suggested (not proved) result.**

  sup_t |e(t)| ≤ (L_g |λ| H + L_F) L_u δ / (g_min μ).

**What is missing.** A complete proof must treat the discontinuous ũ. It has not been written.

## Euler-B global bound (UNPROVED sketch)

- **Local step.** The identity e^z − 1 − z = z² ∫_0^1 (1 − s) e^{sz} ds gives the local bound; this part is proved in the paper.
- **Global step.** Summing the local bound against the contraction suggests a global O(δ) bound. Its constant would grow with Δ|λ|.
- **Status.** The global argument has not been written.

## Refinement with Δ = τ g (CONJECTURE)

- **Observation.** The recurrence is the exact flow of ḣ = (τ/δ) g(ũ) (λ h + B(ũ) ũ).
- **Conjecture.** Its solution should approach −B(u) u / λ as δ → 0, so the artifact would tend to a nonzero limit.
- **Evidence.** The toy data (FR-E1, Δ = τ g column) are consistent with a growing artifact up to m = 8. They are not a proof of the limit.
