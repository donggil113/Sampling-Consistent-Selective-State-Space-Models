"""Verify the pinned TIDES adapter on CPU: forward, backward, optimizer update, eval mode,
per-step step_scale path, and a timing smoke.  Uses 4 TRAIN trajectories only and an
UNTRAINED model; nothing here is a model result.

Output: results/raw/P1-REAL-01__adapter_check.json
"""

import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402

from run_p1_real_01 import TIDES_DIR, labels_for, load_cfg, make_split  # noqa: E402
from scssm.real import protocol as P  # noqa: E402
from scssm.real import tides_adapter as TA  # noqa: E402


def main():
    cfg, fr = load_cfg()
    torch, tides = TA.load_tides(TIDES_DIR)
    torch.set_num_threads(cfg["budget"]["cpu_threads"])
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    rep = {"status": "ADAPTER CHECK with UNTRAINED weights on 4 train trajectories; not a model result",
           "torch": torch.__version__, "threads": torch.get_num_threads()}
    T, K = cfg["task"]["training_grid"]["T"], cfg["task"]["training_grid"]["K"]
    dt = T / K
    params = P.teacher(fr)
    sp = make_split(cfg)
    ids = sp.train[:4]
    grids = [P.training_grid(P.trajectory_path(i), T, K) for i in ids]
    labs = [labels_for(params, P.trajectory_path(i), g.times)[1:] for i, g in zip(ids, grids)]
    xs = P.norm_stats([v for g in grids for v in g.values])
    ys = P.norm_stats([v for lab in labs for v in lab])
    model = TA.build_model(torch, tides, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    rep["n_parameters"] = n_params
    u, s, y = TA.batch_from_grids(torch, grids, labs, dt, xs, ys)

    # forward
    model.train()
    out = TA.forward(torch, model, u, s)
    rep["forward_shape"] = list(out.shape)
    rep["forward_finite"] = bool(torch.isfinite(out).all())
    # backward
    loss = torch.mean((out - y) ** 2)
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters()}
    rep["loss_before"] = float(loss.detach())
    rep["params_with_grad"] = sum(1 for g in grads.values() if g is not None)
    rep["params_total"] = len(grads)
    rep["params_without_grad"] = [n for n, g in grads.items() if g is None]
    rep["grads_finite"] = all(bool(torch.isfinite(g).all()) for g in grads.values() if g is not None)
    rep["grad_norm"] = float(torch.sqrt(sum((g ** 2).sum() for g in grads.values() if g is not None)))
    rep["params_with_all_zero_grad_at_init"] = [n for n, g in grads.items()
                                                if g is not None and float(g.abs().max()) == 0.0]
    rep["explanations"] = {
        "params_without_grad": "static Lambda_re, B, C are not used when lambda_re_mode and bc_mode are "
                               "'input_dependent' (the projection heads with HiPPO biases replace them)",
        "params_with_all_zero_grad_at_init": "the heads' output projections are zero-initialized "
                                             "(proj_init_method='zeros', TIDES Sec. 3.2), so parameters upstream "
                                             "of them receive zero gradient at step 0 only",
    }
    # optimizer update
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    opt.step()
    changed = [n for n, p in model.named_parameters() if not torch.equal(p.detach(), before[n])]
    rep["params_changed_by_step"] = len(changed)
    opt.zero_grad()
    with torch.no_grad():
        rep["loss_after_one_step_same_batch"] = float(torch.mean((TA.forward(torch, model, u, s) - y) ** 2))
    # eval mode determinism and scalar-vs-tensor step_scale path on the training grid
    model.eval()
    with torch.no_grad():
        o1 = TA.forward(torch, model, u, s)
        o2 = TA.forward(torch, model, u, s)
        o3 = model["head"](model["backbone"](u.unsqueeze(-1), step_scale=1.0)).squeeze(-1)
    rep["eval_repeat_max_abs_diff"] = float((o1 - o2).abs().max())
    rep["tensor_vs_scalar_step_max_abs_diff"] = float((o1 - o3).abs().max())
    # timing smoke: training steps and eval on the longest test condition (S8, L = 8K)
    bs = cfg["training"]["batch_size"]
    grids_b = (grids * ((bs + 3) // 4))[:bs]
    labs_b = (labs * ((bs + 3) // 4))[:bs]
    ub, sb, yb = TA.batch_from_grids(torch, grids_b, labs_b, dt, xs, ys)
    model.train()
    t0 = time.time()
    n_steps = 20
    for _ in range(n_steps):
        loss = torch.mean((TA.forward(torch, model, ub, sb) - yb) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
    per_step = (time.time() - t0) / n_steps
    rep["train_seconds_per_step_batch32"] = per_step
    rep["projected_train_seconds_for_configured_steps"] = per_step * cfg["training"]["steps"]
    s8 = [split for split in (P.split_grid(g, 8) for g in grids)]
    u8, st8, _ = TA.batch_from_grids(torch, s8, None, dt, xs, ys)
    _, secs = TA.predict(torch, model, u8, st8)
    rep["eval_seconds_batch4_L256"] = secs
    path = os.path.join(ROOT, "results", "raw", "P1-REAL-01__adapter_check.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
