"""EXPLORATORY (post hoc, added after seeing the P1-REAL-01 results): where in the
frozen TIDES checkpoint does the splitting discrepancy appear?

Same frozen checkpoint (hash-checked) and the same test trajectories; for the
same-held-path refinements S2/S4/S8/H8 it measures, at the common base times,
rel_l2(features(condition), features(C0)) after the encoder, after block 1, after
block 2 and at the output, in float32 and in float64 (weights cast to double).
Output: results/raw/P1-REAL-01__layer_diagnostic.json
"""

import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from run_p1_real_01 import CKPT, RAW, TRAIN_JSON, load_cfg, make_split, seed_all, setup, sha256_file, test_conditions  # noqa: E402
from scssm.real import protocol as P  # noqa: E402


def feats(torch, model, u, s):
    bb = model["backbone"]
    out = {}
    x = bb.encoder(u.unsqueeze(-1))
    out["encoder"] = x
    for b, blk in enumerate(bb.blocks):
        x = blk(x, step_scale=s)
        out[f"block{b + 1}"] = x
    out["output"] = model["head"](x)
    return out


def rel(torch, a, b):
    # per-trajectory relative l2 over (base times x channels), then mean and max
    num = torch.sqrt(((a - b) ** 2).sum(dim=(1, 2)))
    den = torch.sqrt((b ** 2).sum(dim=(1, 2))).clamp_min(1e-300)
    r = num / den
    return float(r.mean()), float(r.max())


def main():
    cfg, fr = load_cfg()
    TA, torch, tides = setup(cfg)
    trn = json.load(open(TRAIN_JSON))
    assert sha256_file(CKPT) == trn["checkpoint_sha256"], "checkpoint hash mismatch"
    x_stats, y_stats = tuple(trn["x_stats"]), tuple(trn["y_stats"])
    seed_all(torch, 0)
    m32 = TA.build_model(torch, tides, cfg)
    m32.load_state_dict(torch.load(CKPT))
    m32.eval()
    m64 = copy.deepcopy(m32).double()
    m64.eval()
    T, K = cfg["task"]["training_grid"]["T"], cfg["task"]["training_grid"]["K"]
    dt = T / K
    tids = list(make_split(cfg).test)
    conds = {i: test_conditions(P.trajectory_path(i), T, K, i) for i in tids}
    res = {"status": "EXPLORATORY post-hoc diagnostic on the frozen seed-0 checkpoint", "conditions": {}}
    for dtype, model in (("float32", m32), ("float64", m64)):
        base_feats = None
        for name in ("C0", "S2", "S4", "S8", "H8"):
            grids = [conds[i][name] for i in tids]
            u, s, _ = TA.batch_from_grids(torch, grids, None, dt, x_stats, y_stats)
            if dtype == "float64":
                u, s = u.double(), s.double()
            with torch.no_grad():
                f = feats(torch, model, u, s)
            idx = torch.tensor([k - 1 for k in grids[0].base_index[1:]])
            f = {k: v[:, idx, :] for k, v in f.items()}
            if name == "C0":
                base_feats = f
                continue
            res["conditions"].setdefault(name, {})[dtype] = {
                k: dict(zip(("mean", "max"), rel(torch, f[k], base_feats[k]))) for k in f}
    path = os.path.join(RAW, "P1-REAL-01__layer_diagnostic.json")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=2)
    for name, d in res["conditions"].items():
        for dtype, v in d.items():
            print(name, dtype, {k: f"{x['mean']:.2e}" for k, x in v.items()})


if __name__ == "__main__":
    main()
