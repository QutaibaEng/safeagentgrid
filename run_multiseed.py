"""THE Independent TRAINING seeds.............

Repeats the COMPLETE training pipeline    (mediator-in-the-loop policy training,
surrogate imitation, conformal train-to-qualify, 8-bit quantisation and
re-calibration) for every training seed, and evaluates every method on the SAME
40 evaluation episodes (evaluation seed 42).  All of the differences between seeds are
therefore due to training stochasticity..........


Per training seed s the following are trained / evaluated:
  SAG     SafeAgentGrid                       normal / 20% FDIA / Byzantine
  greedy  No Mediator (SAG agents, raw)       normal / 20% FDIA
  proj    Projection-only safety layer        normal / 20% FDIA
  ind     Independent PG MARL (CTDE critic)   normal / 20% FDIA   (seed s)
  sac     Single-Agent Constitutional         normal / 20% FDIA   (seed s)
  shield  Joint safety shield                 normal / 20% FDIA   (seed s)
  lag     Lagrangian constrained PG MARL      normal / 20% FDIA   (seed s)
Seed 42 of SafeAgentGrid is bit-identical to the reference model of the paper.



Usage:  python run_multiseed.py --seeds 42 43 44 45 46
        python run_multiseed.py --seeds 47 48 49 50 51      (second worker)
Resumable: one JSON per seed in results/multiseed/.
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
sys.argv = ["ms"]
spec.loader.exec_module(v1)
sys.argv = _argv



ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
ap.add_argument("--episodes", type=int, default=40)
ap.add_argument("--train_eps", type=int, default=30)
ap.add_argument("--N", type=int, default=10)
args = ap.parse_args()

N, E, TE = args.N, args.episodes, args.train_eps
OUTDIR = os.path.join(HERE, "results", "multiseed")
os.makedirs(OUTDIR, exist_ok=True)


def pack(r):
    ep = np.asarray(r["ep_violations"], float) / 200.0
    out = {"avg_reward": r["avg_reward"],
           "violations_per_step": r["violations_per_step"],
           "viol_sem": float(ep.std(ddof=1) / np.sqrt(len(ep))) if len(ep) > 1 else 0.0,
           "efficiency": r["efficiency"], "alignment": r["alignment"],
           "fairness_gini": r["fairness_gini"], "latency_ms": r["latency_ms"],
           "robustness": r["robustness"],
           "fast_share": r.get("fast_share_eval", float("nan")),
           "ep_rewards": r["ep_rewards"], "ep_viol": ep.tolist()}
    for k in ("viol_shortfall", "viol_voltage", "viol_line"):
        if k in r:
            out[k] = r[k]
    return out



for sd in args.seeds:
    fn = os.path.join(OUTDIR, f"seed_{sd}.json")
    R = json.load(open(fn)) if os.path.exists(fn) else {}

    def save():
        with open(fn, "w") as fh:
            json.dump(R, fh, indent=1)

    def stage(key, f):
        if key in R:
            return
        t0 = time.time()
        R[key] = pack(f())
        R[key]["wall_s"] = time.time() - t0
        print(f"seed {sd} {key:16s} rew {R[key]['avg_reward']:10.1f} viol "
              f"{R[key]['violations_per_step']:.4f} +/-{R[key]['viol_sem']:.4f} "
              f"eff {R[key]['efficiency']:.2f} gini {R[key]['fairness_gini']:.4f} "
              f"lat {R[key]['latency_ms']:.2f} ({R[key]['wall_s']:.0f}s)", flush=True)
        save()

    # ---###- SafeAgentGrid: the three evaluations are always run together and in
    # ======  the paper's order, because the mediator carries state between them.
    sag_keys = ("SAG_normal", "SAG_fdia", "SAG_byz", "greedy_normal",
                "greedy_fdia", "proj_normal", "proj_fdia")
    if any(k not in R for k in sag_keys):
        for k in sag_keys:
            R.pop(k, None)
        t0 = time.time()
        med, ags, curve = v1.train_safeagentgrid(TE, N, 40, verbose=False,
                                                 surr_epochs=200, seed=sd)
        R["training_curve"] = curve
        R["conformal_q"] = med.conformal_q
        R["train_wall_s"] = time.time() - t0
        save()
        stage("SAG_normal", lambda: (med.reset_telemetry(),
                                     v1.evaluate(med, ags, N, E))[1])
        med.solver_log = []
        stage("SAG_fdia", lambda: (med.reset_telemetry(),
                                   v1.evaluate(med, ags, N, E, attack="fdia"))[1])
        sl = np.asarray(med.solver_log, dtype=float)
        R["slsqp_fdia"] = {"n": int(len(sl)),
                           "success_rate": float(sl[:, 0].mean()),
                           "mean_iter": float(sl[:, 1].mean()),
                           "max_iter": float(sl[:, 1].max()),
                           "at_cap_rate": float((sl[:, 1] >= 40).mean())}
        med.solver_log = None
        save()
        stage("SAG_byz", lambda: (med.reset_telemetry(),
                                  v1.evaluate(med, ags, N, E, attack="byzantine"))[1])
        stage("greedy_normal", lambda: v1.evaluate_policy_only(ags, N, E, mode="greedy"))
        stage("greedy_fdia", lambda: v1.evaluate_policy_only(ags, N, E, mode="greedy",
                                                             attack="fdia"))
        stage("proj_normal", lambda: v1.evaluate_policy_only(ags, N, E, mode="projection"))
        stage("proj_fdia", lambda: v1.evaluate_policy_only(ags, N, E, mode="projection",
                                                           attack="fdia"))

    # ---======#####- unmediated baselines (trained with the same seed)
    if "ind_normal" not in R or "ind_fdia" not in R:
        ag = v1.train_unmediated(TE, N, shared_policy=False, constitutional=False, seed=sd)
        R.pop("ind_normal", None)
        stage("ind_normal", lambda: v1.evaluate_policy_only(ag, N, E, mode="greedy"))
        stage("ind_fdia", lambda: v1.evaluate_policy_only(ag, N, E, mode="greedy",
                                                          attack="fdia"))
    if "sac_normal" not in R or "sac_fdia" not in R:
        ag = v1.train_unmediated(TE, N, shared_policy=True, constitutional=True, seed=sd)
        R.pop("sac_normal", None)
        stage("sac_normal", lambda: v1.evaluate_policy_only(ag, N, E, mode="projection"))
        stage("sac_fdia", lambda: v1.evaluate_policy_only(ag, N, E, mode="projection",
                                                          attack="fdia"))

    # --############-- joint network-level safety shield and Lagrangian constrained MARL
    if "shield_normal" not in R or "shield_fdia" not in R:
        ag = v1.train_unmediated(TE, N, shared_policy=False, constitutional=False,
                                 seed=sd, shield=v1.JointSafetyShield())
        R.pop("shield_normal", None)
        stage("shield_normal", lambda: v1.evaluate_policy_only(
            ag, N, E, mode="shield", shield=v1.JointSafetyShield()))
        stage("shield_fdia", lambda: v1.evaluate_policy_only(
            ag, N, E, mode="shield", attack="fdia", shield=v1.JointSafetyShield()))
    if "lag_normal" not in R or "lag_fdia" not in R:
        ag = v1.train_unmediated(TE, N, shared_policy=False, constitutional=False,
                                 seed=sd, lagrangian=True, cost_limit=0.0, lag_lr=50.0)
        R.pop("lag_normal", None)
        stage("lag_normal", lambda: v1.evaluate_policy_only(ag, N, E, mode="greedy"))
        stage("lag_fdia", lambda: v1.evaluate_policy_only(ag, N, E, mode="greedy",
                                                          attack="fdia"))
    print(f"SEED {sd} COMPLETE", flush=True)
print("MULTISEED COMPLETE", flush=True)
