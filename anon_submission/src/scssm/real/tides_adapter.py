"""Adapter for the official TIDES PyTorch implementation (P1-REAL-01).

STATUS: NOT_RUN. torch is not installed and its installation is not approved;
the official code is not vendored.  This module is written so that the run is a
mechanical step once the approval items in configs/p1_real_01.json are granted.

Pinned implementation: github.com/TaylanSoydan/TIDES @ 4b51adce2060e7209e002a6a2fd6691a2f6fcc5e
(MIT).  Expected checkout location: third_party/TIDES (see configs/p1_real_01.json).

Nothing here may be used to report trained-model results unless the training
stage actually ran; results from untrained (randomly initialized) weights must be
labelled UNTRAINED.
"""

import os
import sys
import time

TIDES_COMMIT = "4b51adce2060e7209e002a6a2fd6691a2f6fcc5e"


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment dependent
        raise RuntimeError("NOT_RUN: torch is not installed (installation requires approval)") from e
    return torch


def load_tides(repo_dir):
    """Import the pinned official TIDES package from a local checkout."""
    torch = _require_torch()
    head = os.path.join(repo_dir, ".git", "HEAD")
    if not os.path.exists(os.path.join(repo_dir, "tides", "tides.py")):
        raise RuntimeError(f"NOT_RUN: TIDES checkout not found at {repo_dir}")
    if os.path.exists(head):
        import subprocess
        rev = subprocess.check_output(["git", "-C", repo_dir, "rev-parse", "HEAD"], text=True).strip()
        if rev != TIDES_COMMIT:
            raise RuntimeError(f"TIDES checkout is at {rev}, expected pinned {TIDES_COMMIT}")
    sys.path.insert(0, repo_dir)
    import tides  # noqa: E402  (official package)
    return torch, tides


def build_model(torch, tides, cfg):
    """Backbone = official TIDES; head = one linear map per time step."""
    m = cfg["model"]
    backbone = tides.TIDES(
        d_input=1, d_hidden=m["d_hidden"], ssm_size=m["ssm_size"], ssm_blocks=m["ssm_blocks"],
        num_blocks=m["num_blocks"], discretization="zoh", lambda_re_mode=m["lambda_re_mode"],
        lambda_im_mode=m["lambda_im_mode"], bc_mode=m["bc_mode"], bidir=False,
        conv_kernel_size=0, drop_rate=m["drop_rate"],
    )
    head = torch.nn.Linear(m["d_hidden"], 1)
    return torch.nn.ModuleDict({"backbone": backbone, "head": head})


def forward(torch, model, u, step_scale):
    """u: (B, L) normalized inputs; step_scale: (B, L) physical gaps / dt_train."""
    h = model["backbone"](u.unsqueeze(-1), step_scale=step_scale)
    return model["head"](h).squeeze(-1)


def batch_from_grids(torch, grids, targets, dt_train, x_stats, y_stats):
    """Stack same-length grids into tensors (all grids in one batch share L)."""
    from .protocol import step_scales
    mu_x, sd_x = x_stats
    mu_y, sd_y = y_stats
    u = torch.tensor([[(v - mu_x) / sd_x for v in g.values] for g in grids], dtype=torch.float32)
    s = torch.tensor([step_scales(g, dt_train) for g in grids], dtype=torch.float32)
    y = None if targets is None else torch.tensor([[(v - mu_y) / sd_y for v in t] for t in targets],
                                                  dtype=torch.float32)
    return u, s, y


def train(torch, model, batches, cfg, log):
    """Fixed-step Adam training; no schedule, no early stopping, no sweep."""
    tr = cfg["training"]
    torch.manual_seed(tr["seed"])
    torch.set_num_threads(cfg["budget"]["cpu_threads"])
    opt = torch.optim.Adam(model.parameters(), lr=tr["lr"])
    model.train()
    t0 = time.time()
    for step in range(tr["steps"]):
        u, s, y = batches(step)
        loss = torch.mean((forward(torch, model, u, s) - y) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % tr["log_every"] == 0:
            log(f"step {step} loss {loss.item():.6f} elapsed {time.time() - t0:.1f}s")
        if time.time() - t0 > cfg["budget"]["train_wall_clock_seconds"]:
            log(f"ABORT: training budget exceeded at step {step} -> NOT_RUN(TIMEOUT); no results may be reported")
            return False
    return True


def predict(torch, model, u, s, repeats=3):
    """Eval-mode prediction and median wall time per call (inference cost)."""
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = forward(torch, model, u, s)
            times.append(time.perf_counter() - t0)
    times.sort()
    return out, times[len(times) // 2]
