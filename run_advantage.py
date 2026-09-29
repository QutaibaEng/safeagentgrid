"""THe robustness on the utilized threat models (Table 8): SafeAgentGrid, SafeAgentGrid
without its security module, and the joint safety shield under the conditions that
Table 6 does not cover......

Everything utilizes the training seed 42 (the reference model) and the same 40
evaluation episodes (evaluation seed 42) as the rest of the paper.
  part A  SafeAgentGrid (and its security-off variant) under
            - Byzantine failures, stealth FDIA U[1,2]            (paper's threat models)
            - network-targeted FDIA (apply_fdia_flow), |delta| in U[20,35], U[2,5], U[1,2]


  part B  the joint safety shield under the same conditions
All of the results are reported, whichever method they favour...

Usage: python run_advantage.py --part A|B     (resumable; results/advantage_{part}.json)
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
sys.argv = ["adv"]
spec.loader.exec_module(v1)
sys.argv = _argv




ap = argparse.ArgumentParser()
ap.add_argument("--part", required=True, choices=["A", "B"])
args = ap.parse_args()

OUT = os.path.join(HERE, "results", f"advantage_{args.part}.json")
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
E = 40


def save():
    json.dump(R, open(OUT, "w"), indent=1)





def pack(r):
    ep = np.asarray(r["ep_violations"], float) / 200.0
    out = {"avg_reward": r["avg_reward"], "violations_per_step": r["violations_per_step"],
           "viol_sem": float(ep.std(ddof=1) / np.sqrt(len(ep))),
           "efficiency": r["efficiency"], "fairness_gini": r["fairness_gini"],
           "robustness": r["robustness"], "alignment": r["alignment"],
           "latency_ms": r["latency_ms"], "detect_rate": r.get("anomaly_rate_eval"),
           "ep_viol": ep.tolist(), "ep_rewards": r["ep_rewards"]}
    for k in ("viol_line", "viol_shortfall", "viol_voltage"):
        if k in r:
            out[k] = r[k]
    return out


def stage(key, f):
    if key in R:
        return
    t0 = time.time()
    R[key] = pack(f())
    x = R[key]
    print(f"{key:28s} viol {x['violations_per_step']:.4f} +/- {x['viol_sem']:.4f} "
          f"rew {x['avg_reward']:9.1f} gini {x['fairness_gini']:.4f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    save()


ATTACKS = [("fdia", {"attack": "fdia"}),
           ("byz", {"attack": "byzantine"}),
           ("stealth12", {"attack": "fdia", "fdia_lo": 1.0, "fdia_hi": 2.0}),
           ("flow_20_35", {"attack": "fdia_flow"}),
           ("flow_2_5", {"attack": "fdia_flow", "fdia_lo": 2.0, "fdia_hi": 5.0}),
           ("flow_1_2", {"attack": "fdia_flow", "fdia_lo": 1.0, "fdia_hi": 2.0})]

if args.part == "A":
    N = 10
    med, ags, _ = v1.train_safeagentgrid(30, N, 40, verbose=False, surr_epochs=200, seed=42)
    # same order as the paper (the mediator carries state between evaluations)
    stage("SAG_normal", lambda: (med.reset_telemetry(), v1.evaluate(med, ags, N, E))[1])
    for name, kw in ATTACKS:
        stage(f"SAG_{name}", lambda: (med.reset_telemetry(),
                                       v1.evaluate(med, ags, N, E, **kw))[1])
    nosec = v1.SafeAgentGridMediator(N, 2 * N + 1, security_enabled=False)
    nosec.fast_surrogate = med.fast_surrogate
    nosec.conformal_q = med.conformal_q
    nosec.surrogate_trained = True
    for name, kw in ATTACKS:
        stage(f"SAGnosec_{name}", lambda: (nosec.reset_telemetry(),
                                            v1.evaluate(nosec, ags, N, E, **kw))[1])



else:
    N = 10
    ag = v1.train_unmediated(30, N, shared_policy=False, constitutional=False, seed=42,
                             shield=v1.JointSafetyShield())
    stage("shield_normal", lambda: v1.evaluate_policy_only(
        ag, N, E, mode="shield", shield=v1.JointSafetyShield()))
    for name, kw in ATTACKS:
        stage(f"shield_{name}", lambda: v1.evaluate_policy_only(
            ag, N, E, mode="shield", shield=v1.JointSafetyShield(), **kw))



print(f"PART {args.part} COMPLETE", flush=True)
