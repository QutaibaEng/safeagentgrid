"""NOW How are lambda, mu, nu and rho_t coordinated, and how does the
full-path program prioritise conflicting objectives??????

One-at-a-time sensitivity of the full-path objective weights on the reference
model (training seed 42), 40 evaluation episodes per setting:


  lambda (constitutional distance / reserve), mu (tightened reachability),
  nu (fairness, via nu_scale)                    -> nominal conditions
  rho_t (security penalty, via rho_scale)        -> 20% FDIA (it only acts on
                                                    flagged agents)
The fast path is DISABLED so that every decision is taken by the program whose
weights are varied (the surrogate imitates the default weights).  The default
row (all multipliers = 1) equals the "w/o hybrid fast path" ablation.

Usage: python run_weight_sensitivity.py     (resumable; results/weight_sensitivity.json)
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
sys.argv = ["w"]
spec.loader.exec_module(v1)

OUT = os.path.join(HERE, "results", "weight_sensitivity.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}





def save():
    json.dump(R, open(OUT, "w"), indent=1)


def pack(r):
    ep = np.asarray(r["ep_violations"], float) / 200.0
    return {"avg_reward": r["avg_reward"], "violations_per_step": r["violations_per_step"],
            "viol_sem": float(ep.std(ddof=1) / np.sqrt(len(ep))),
            "viol_line": r.get("viol_line"), "viol_shortfall": r.get("viol_shortfall"),
            "efficiency": r["efficiency"], "alignment": r["alignment"],
            "fairness_gini": r["fairness_gini"], "latency_ms": r["latency_ms"],
            "robustness": r["robustness"], "ep_viol": ep.tolist(),
            "ep_rewards": r["ep_rewards"]}


E = 40
PLAN = [("base", 1.0, "normal")]
PLAN += [("lam", m, "normal") for m in (0.1, 0.5, 2.0, 10.0)]
PLAN += [("mu", m, "normal") for m in (0.1, 0.5, 2.0, 10.0)]
PLAN += [("nu", m, "normal") for m in (0.1, 0.5, 2.0, 10.0, 100.0)]
PLAN += [("rho", m, "fdia") for m in (0.0, 0.1, 1.0, 10.0, 100.0)]

if any(f"{w}{m}_{c}" not in R for w, m, c in PLAN):
    med, ags, _ = v1.train_safeagentgrid(30, 10, 40, verbose=False, surr_epochs=200, seed=42)
    med.fast_path_enabled = False
    LAM0, MU0 = med.LAM, med.MU
    for w, m, cond in PLAN:
        key = f"{w}{m}_{cond}"
        if key in R:
            continue
        med.LAM, med.MU, med.nu_scale, med.rho_scale = LAM0, MU0, 1.0, 1.0
        if w == "lam":
            med.LAM = LAM0 * m
        elif w == "mu":
            med.MU = MU0 * m
        elif w == "nu":
            med.nu_scale = m
        elif w == "rho":
            med.rho_scale = m
        med.reset_telemetry()
        t0 = time.time()
        r = v1.evaluate(med, ags, 10, E, attack=None if cond == "normal" else "fdia")
        R[key] = pack(r)
        R[key]["weights"] = {"lambda": med.LAM, "mu": med.MU,
                             "nu": 0.3 * med.nu_scale, "rho_scale": med.rho_scale}
        print(f"{key:14s} viol {r['violations_per_step']:.4f} rew {r['avg_reward']:9.1f} "
              f"gini {r['fairness_gini']:.4f} eff {r['efficiency']:.2f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        save()
    med.LAM, med.MU, med.nu_scale, med.rho_scale = LAM0, MU0, 1.0, 1.0

print("WEIGHT SENSITIVITY COMPLETE", flush=True)
