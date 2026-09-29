"""THE sensitivity of the training reward of Eq. (27) of the paper,
    r_t^total = r_t - beta * Gini_t - lambda_sec * ||a'_t - a*_t||^2 / N,
to its two weights.  One-at-a-time retraining with training seed 42:
    beta       in {0, 0.3 (default), 1.0}
    lambda_sec in {0, 0.01 (default), 0.1}
Each model is evaluated on the same 40 episodes (nominal and 20% FDIA).  The
effect sizes should be read against the seed-to-seed spread of
run_multiseed.py......

Usage: python run_training_weights.py      (results/training_weights.json)
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("v1", os.path.join(HERE, "safeagentgrid.py"))
v1 = importlib.util.module_from_spec(spec)
sys.argv = ["tw"]
spec.loader.exec_module(v1)

OUT = os.path.join(HERE, "results", "training_weights.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(R, open(OUT, "w"), indent=1)


def pack(r):
    ep = np.asarray(r["ep_violations"], float) / 200.0
    return {"avg_reward": r["avg_reward"], "violations_per_step": r["violations_per_step"],
            "viol_sem": float(ep.std(ddof=1) / np.sqrt(len(ep))),
            "efficiency": r["efficiency"], "alignment": r["alignment"],
            "fairness_gini": r["fairness_gini"], "latency_ms": r["latency_ms"],
            "robustness": r["robustness"], "fast_share": r.get("fast_share_eval"),
            "ep_viol": ep.tolist(), "ep_rewards": r["ep_rewards"]}


N, E = 10, 40
PLAN = [("beta", 0.0), ("beta", 1.0), ("lamsec", 0.0), ("lamsec", 0.1)]
for w, val in PLAN:
    key = f"{w}{val}"
    if key in R:
        continue
    kw = {"beta": val} if w == "beta" else {"lam_sec": val}
    t0 = time.time()
    med, ags, curve = v1.train_safeagentgrid(30, N, 40, verbose=False, surr_epochs=200,
                                             seed=42, **kw)
    res = {"training_curve": curve, "conformal_q": med.conformal_q,
           "train_wall_s": time.time() - t0}
    med.reset_telemetry()
    res["normal"] = pack(v1.evaluate(med, ags, N, E))
    med.reset_telemetry()
    res["fdia"] = pack(v1.evaluate(med, ags, N, E, attack="fdia"))
    R[key] = res
    print(f"{key:10s} normal viol {res['normal']['violations_per_step']:.4f} rew "
          f"{res['normal']['avg_reward']:9.1f} gini {res['normal']['fairness_gini']:.4f} | "
          f"fdia viol {res['fdia']['violations_per_step']:.4f} | q {med.conformal_q:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    save()





print("TRAINING WEIGHTS COMPLETE", flush=True)
