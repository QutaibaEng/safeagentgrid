"""THE Ten-seed confirmation of the robustness comparison of
run_advantage.py (Table 8) under the Byzantine failures and the network-targeted FDIA
(apply_fdia_flow, |delta| ~ U[20,35]). THE  Same protocol as run_multiseed.py: the
complete training pipeline is repeated for every training seed, and every trained
model has been evaluated on the same 40 evaluation episodes.

  --method SAG     SafeAgentGrid: normal, 20% FDIA, Byzantine (reproduce the values
                   of results/multiseed/ for the same seed), then network-targeted FDIA
  --method shield  joint safety shield: normal, 20% FDIA, Byzantine,
                   network-targeted FDIA
Usage: python run_advantage_ms.py --method SAG --seeds 43 44 45 46 47
Resumable: results/advantage_ms/{method}_seed_{s}.json
"""


import argparse
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
_argv = sys.argv
sys.argv = ["advms"]
spec.loader.exec_module(v1)
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument("--method", required=True, choices=["SAG", "shield"])
ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
args = ap.parse_args()
N, E, TE = 10, 40, 30
OUTDIR = os.path.join(HERE, "results", "advantage_ms")
os.makedirs(OUTDIR, exist_ok=True)


def pack(r):
    ep = np.asarray(r["ep_violations"], float) / 200.0
    return {"avg_reward": r["avg_reward"], "violations_per_step": r["violations_per_step"],
            "viol_sem": float(ep.std(ddof=1) / np.sqrt(len(ep))),
            "efficiency": r["efficiency"], "fairness_gini": r["fairness_gini"],
            "robustness": r["robustness"], "ep_viol": ep.tolist(),
            "ep_rewards": r["ep_rewards"]}


for sd in args.seeds:
    fn = os.path.join(OUTDIR, f"{args.method}_seed_{sd}.json")
    R = json.load(open(fn)) if os.path.exists(fn) else {}

    def stage(key, f):
        if key in R:
            return
        t0 = time.time()
        R[key] = pack(f())
        print(f"seed {sd} {key:24s} viol {R[key]['violations_per_step']:.4f} "
              f"+/- {R[key]['viol_sem']:.4f} rew {R[key]['avg_reward']:9.1f} "
              f"gini {R[key]['fairness_gini']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(R, open(fn, "w"), indent=1)

    if args.method == "SAG":
        keys = ("SAG_normal", "SAG_fdia", "SAG_byz", "SAG_flow")
        if any(k not in R for k in keys):
            for k in keys:                      # the mediator carries state: rerun in order
                R.pop(k, None)
            med, ags, _ = v1.train_safeagentgrid(TE, N, 40, verbose=False, surr_epochs=200,
                                                 seed=sd)
            stage("SAG_normal", lambda: (med.reset_telemetry(), v1.evaluate(med, ags, N, E))[1])
            stage("SAG_fdia", lambda: (med.reset_telemetry(),
                                       v1.evaluate(med, ags, N, E, attack="fdia"))[1])
            stage("SAG_byz", lambda: (med.reset_telemetry(),
                                      v1.evaluate(med, ags, N, E, attack="byzantine"))[1])
            stage("SAG_flow", lambda: (med.reset_telemetry(),
                                       v1.evaluate(med, ags, N, E, attack="fdia_flow"))[1])
    elif any(f"shield_{c}" not in R for c in ("normal", "fdia", "byz", "flow")):
        ag = v1.train_unmediated(TE, N, shared_policy=False, constitutional=False,
                                 seed=sd, shield=v1.JointSafetyShield())
        for c, kw in (("normal", {}), ("fdia", {"attack": "fdia"}),
                      ("byz", {"attack": "byzantine"}), ("flow", {"attack": "fdia_flow"})):
            R.pop(f"shield_{c}", None)
            stage(f"shield_{c}", lambda: v1.evaluate_policy_only(
                ag, N, E, mode="shield", shield=v1.JointSafetyShield(), **kw))
    print(f"SEED {sd} COMPLETE", flush=True)


print("ADVANTAGE_MS COMPLETE", flush=True)
