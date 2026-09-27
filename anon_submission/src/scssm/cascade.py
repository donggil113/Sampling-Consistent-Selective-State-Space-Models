"""P1-COMP-01: two-layer linear cascade under interval splitting.

    dh1/dt = -a*h1 + b*u(t)
    dh2/dt = -c*h2 + d*h1(t)

with u held (right-endpoint convention) on each grid interval.  Three
computations are compared at the common base times:

* coupled_exact : closed-form flow of the 2x2 system over each held interval.
* coupled_rk4   : independent classical RK4 of the coupled system.
* layerwise     : layer 1 exact ZOH; layer 2 exact ZOH driven by the layer-1
                  endpoint value h1(s_j) held on (s_{j-1}, s_j].

This is a check of standard linear-ODE properties, not a new architecture.
"""

import math

from .model import phi1


def _real(z):
    return complex(z).real


def coupled_step(h1, h2, u, tau, a, b, c, d):
    """Exact flow over one held interval of length tau.

    h1(tau) = e^{-a tau} h1 + b u tau phi1(-a tau)
    h2(tau) = e^{-c tau} h2 + d h1 psi + d b u chi, with
      psi = int_0^tau e^{-c(tau-s)} e^{-a s} ds = e^{-c tau} tau phi1((c-a) tau)
      chi = int_0^tau e^{-c(tau-s)} (1 - e^{-a s})/a ds = (tau phi1(-c tau) - psi)/a
    phi1 is continuous at 0, so a = c (Jordan block, psi = tau e^{-a tau}) needs
    no special case.  chi loses relative accuracy ~ eps/(a tau) when a*tau -> 0.
    """
    ea = math.exp(-a * tau)
    ec = math.exp(-c * tau)
    psi = ec * tau * _real(phi1((c - a) * tau))
    chi = (tau * _real(phi1(-c * tau)) - psi) / a
    h1n = ea * h1 + b * u * tau * _real(phi1(-a * tau))
    h2n = ec * h2 + d * h1 * psi + d * b * u * chi
    return h1n, h2n


def run_coupled_exact(times, values, a, b, c, d):
    h1, h2 = 0.0, 0.0
    H1, H2 = [h1], [h2]
    for k in range(1, len(times)):
        h1, h2 = coupled_step(h1, h2, values[k - 1], times[k] - times[k - 1], a, b, c, d)
        H1.append(h1)
        H2.append(h2)
    return H1, H2


def run_coupled_rk4(times, values, a, b, c, d, n_sub):
    """Independent RK4 on the coupled system with the held input."""

    def f(x1, x2, u):
        return -a * x1 + b * u, -c * x2 + d * x1

    h1, h2 = 0.0, 0.0
    H1, H2 = [h1], [h2]
    for k in range(1, len(times)):
        u = values[k - 1]
        hstep = (times[k] - times[k - 1]) / n_sub
        for _ in range(n_sub):
            k1 = f(h1, h2, u)
            k2 = f(h1 + 0.5 * hstep * k1[0], h2 + 0.5 * hstep * k1[1], u)
            k3 = f(h1 + 0.5 * hstep * k2[0], h2 + 0.5 * hstep * k2[1], u)
            k4 = f(h1 + hstep * k3[0], h2 + hstep * k3[1], u)
            h1 += hstep / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
            h2 += hstep / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        H1.append(h1)
        H2.append(h2)
    return H1, H2


def zoh_scalar(times, values, rate, gain):
    """Exact ZOH of dh/dt = -rate*h + gain*v with v held (right endpoint)."""
    h = 0.0
    H = [h]
    for k in range(1, len(times)):
        tau = times[k] - times[k - 1]
        h = math.exp(-rate * tau) * h + gain * values[k - 1] * tau * _real(phi1(-rate * tau))
        H.append(h)
    return H


def run_layerwise(times, values, a, b, c, d):
    """Layer 1 exact; layer 2 sees the held layer-1 endpoint values."""
    H1 = zoh_scalar(times, values, a, b)
    H2 = zoh_scalar(times, H1[1:], c, d)
    return H1, H2
