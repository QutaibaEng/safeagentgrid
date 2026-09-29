"""THE Three sweeps of the paper, all at the uniform 40-episode protocol:
  * stealth-attack sweep   (Table "stealth": delta ~ U[1,2], U[2,5], U[5,10], U[20,35])
  * delegation sweep       (Table "eta": eta in {0.05, 0.5, 2, 10}, security on/off)
  * median- vs reference-anchored repair under 20% FDIA
All on the reference model (training seed 42), 20% of the agents corrupted.

Usage: python run_sweeps40.py        (resumable; results/sweeps40.json)
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
sys.argv = ["s40"]
spec.loader.exec_module(v1)

OUT = os.path.join(HERE, "results", "sweeps40.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(R, open(OUT, "w"), indent=1)


def pack(r):
    ep = np.asarray(r["ep_violations"], float) / 200.0
    return {"avg_reward": r["avg_reward"], "violations_per_step": r["violations_per_step"],
            "viol_sem": float(ep.std(ddof=1) / np.sqrt(len(ep))),
            "robustness": r["robustness"], "efficiency": r["efficiency"],
            "fairness_gini": r["fairness_gini"], "latency_ms": r["latency_ms"],
            "detect_rate": r.get("anomaly_rate_eval"), "ep_viol": ep.tolist(),
            "ep_rewards": r["ep_rewards"]}


N, E = 10, 40
med, ags, _ = v1.train_safeagentgrid(30, N, 40, verbose=False, surr_epochs=200, seed=42)


def run(key, m, **kw):
    if key in R:
        return
    m.reset_telemetry()
    t0 = time.time()
    R[key] = pack(v1.evaluate(m, ags, N, E, **kw))
    print(f"{key:26s} viol {R[key]['violations_per_step']:.4f} +/- {R[key]['viol_sem']:.4f} "
          f"robust {R[key]['robustness']:.3f} rew {R[key]['avg_reward']:9.1f} "
          f"detect {R[key]['detect_rate']} ({time.time()-t0:.0f}s)", flush=True)
    save()


# --========-- stealth sweep (same order and settings as run_stealth_sweep)
for lo, hi in ((1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (20.0, 35.0)):
    run(f"stealth_{lo:g}_{hi:g}", med, attack="fdia", attack_frac=0.20,
        fdia_lo=lo, fdia_hi=hi)

# -#######- median- vs reference-anchored repair (fresh mediators sharing the surrogate)
def clone(**kw):
    m = v1.SafeAgentGridMediator(N, 2 * N + 1, **kw)
    m.fast_surrogate = med.fast_surrogate
    m.conformal_q = med.conformal_q
    m.surrogate_trained = True
    return m


run("repair_reference_fdia", clone(repair="reference"), attack="fdia")
run("repair_median_fdia", clone(repair="median"), attack="fdia")

# -=======- delegation (eta) sweep, security on / off
mon, moff = clone(), clone(security_enabled=False)
for eta in (0.05, 0.5, 2.0, 10.0):
    for tag, m in (("secured", mon), ("unsecured", moff)):
        m.ETA = eta
        run(f"eta{eta:g}_{tag}", m, attack="fdia")
print("SWEEPS40 COMPLETE", flush=True)
