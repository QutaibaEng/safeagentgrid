"""How was chi = 0.5 determined?  Sweep of the reachability
tightening fraction chi on the reference model (training seed 42)......


The fast path is DISABLED so that every decision is taken by the constrained
program whose feasible set is being tightened; the surrogate was trained to
imitate the chi = 0.5 mediator and would otherwise mask the effect.
Conditions: the nominal and 20% FDIA, 40 evaluation episodes each.

Second part: the analytic chance-constraint rule of the paper,
    chi*(N) = 1 - z * sigma_l(N) / f_max,   sigma_l = sigma_w * ||A_l||_2,
predicts that a fixed chi = 0.5 is too loose at N = 16.  We therefore also
evaluate an N = 16 model (training seed 42) at chi = 0.5 and at chi*(16).

Usage: python run_chi_sweep.py            (resumable; results/chi_sweep.json)
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
from scipy.stats import norm





HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("v1", os.path.join(HERE, "safeagentgrid.py"))
v1 = importlib.util.module_from_spec(spec)
sys.argv = ["chi"]
spec.loader.exec_module(v1)

OUT = os.path.join(HERE, "results", "chi_sweep.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(R, open(OUT, "w"), indent=1)


def pack(r):
    ep = np.asarray(r["ep_violations"], float) / 200.0
    return {"avg_reward": r["avg_reward"], "violations_per_step": r["violations_per_step"],
            "viol_sem": float(ep.std(ddof=1) / np.sqrt(len(ep))),
            "viol_line": r.get("viol_line"), "viol_shortfall": r.get("viol_shortfall"),
            "viol_voltage": r.get("viol_voltage"), "efficiency": r["efficiency"],
            "alignment": r["alignment"], "fairness_gini": r["fairness_gini"],
            "latency_ms": r["latency_ms"], "robustness": r["robustness"],
            "ep_viol": ep.tolist(), "ep_rewards": r["ep_rewards"]}


def sigma_line(N, sigma_w=6.0):
    env = v1.SmartGridEnv(N=N)
    return float(sigma_w * np.max(np.linalg.norm(env._A, axis=1)))




E = 40
CHIS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
R["analytic"] = {str(N): {"sigma_line_pu": sigma_line(N),
                          "chi_star_same_risk_as_N10": None} for N in (4, 8, 10, 16)}
z10 = (1 - 0.5) * v1.SmartGridEnv.F_MAX / sigma_line(10)   # risk level implied by chi=0.5 @ N=10
for N in (4, 8, 10, 16):
    R["analytic"][str(N)]["chi_star_same_risk_as_N10"] = float(
        1 - z10 * sigma_line(N) / v1.SmartGridEnv.F_MAX)
R["analytic"]["z_implied_by_chi0.5_N10"] = float(z10)
R["analytic"]["per_line_risk_implied"] = float(norm.sf(z10))
save()

need = [c for c in CHIS for cond in ("normal", "fdia") if f"chi{c}_{cond}" not in R]
if need:
    med, ags, _ = v1.train_safeagentgrid(30, 10, 40, verbose=False, surr_epochs=200, seed=42)
    med.fast_path_enabled = False
    for c in CHIS:
        for cond in ("normal", "fdia"):
            key = f"chi{c}_{cond}"
            if key in R:
                continue
            med.TIGHTEN = c
            med.reset_telemetry()
            t0 = time.time()
            r = v1.evaluate(med, ags, 10, E, attack=None if cond == "normal" else "fdia")
            R[key] = pack(r)
            print(f"chi={c:.1f} {cond:6s} viol {r['violations_per_step']:.4f} "
                  f"(line {r['viol_line']:.4f}, short {r['viol_shortfall']:.4f}) "
                  f"rew {r['avg_reward']:9.1f} gini {r['fairness_gini']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            save()


# --#######-- N = 16: fixed chi = 0.5 versus the size-adjusted chi*(16)
chi16 = round(R["analytic"]["16"]["chi_star_same_risk_as_N10"], 2)
need16 = [k for k in (f"N16_chi0.5", f"N16_chi{chi16}") if k not in R]
if need16:
    med, ags, _ = v1.train_safeagentgrid(30, 16, 40, verbose=False, surr_epochs=200, seed=42)
    med.fast_path_enabled = False
    for c in (0.5, chi16):
        key = f"N16_chi{c}"
        if key in R:
            continue
        med.TIGHTEN = c
        med.reset_telemetry()
        t0 = time.time()
        r = v1.evaluate(med, ags, 16, E)
        R[key] = pack(r)
        print(f"N=16 chi={c:.2f} viol {r['violations_per_step']:.4f} "
              f"(line {r['viol_line']:.4f}) rew {r['avg_reward']:9.1f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        save()
print("CHI SWEEP COMPLETE", flush=True)
