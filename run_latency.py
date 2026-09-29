"""Like-for-like latency benchmark..............

As mention , the latency column of the main table mixes two different quantities: for
SafeAgentGrid it is the MEDIATION time (proposals -> executed action), for the
unmediated baselines it is the time to evaluate all ten policies sequentially.
This script measures, in one process and on one machine, the per-step time of
  (a) policy inference for the ten agents (sequential, 8-bit models),
  (b) SafeAgentGrid mediation (nominal, hybrid fast/full path),
  (c) SafeAgentGrid full path only,
  (d) the joint safety shield projection only,
  (e) the per-agent box projection only.
Run it on an otherwise idle machine.  Usage: python run_latency.py
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
sys.argv = ["lat"]
spec.loader.exec_module(v1)
OUT = os.path.join(HERE, "results", "latency.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

N, E = 10, 10
med, ags, _ = v1.train_safeagentgrid(30, N, 40, verbose=False, surr_epochs=200, seed=42)
shield = v1.JointSafetyShield()
box = v1.Constitution()



def run(mode):
    np.random.seed(42)
    torch.manual_seed(42)
    env = v1.SmartGridEnv(N=N)
    med.security.reset()
    t_inf, t_dec = [], []
    for _ in range(E):
        s = env.reset()
        for _ in range(env.T):
            t0 = time.perf_counter()
            p = [float(ags[i].act(s, deterministic=True)[0]) for i in range(N)]
            t1 = time.perf_counter()
            if mode == "sag":
                a, _, _ = med.mediate(p, s, env)
            elif mode == "shield":
                a = shield.project(np.asarray(p), env)
            else:
                a = box.filter(np.asarray(p))
            t2 = time.perf_counter()
            t_inf.append(1e3 * (t1 - t0))
            t_dec.append(1e3 * (t2 - t1))
            s, _, done, _ = env.step(a)
            if done:
                break
    return float(np.mean(t_inf)), float(np.mean(t_dec)), float(np.percentile(t_dec, 95))


R = {}
for mode in ("sag", "shield", "box"):
    inf, dec, p95 = run(mode)
    R[mode] = {"inference_ms": inf, "decision_ms": dec, "decision_p95_ms": p95}
    print(f"{mode:7s} inference {inf:.3f} ms | decision layer {dec:.3f} ms (p95 {p95:.3f})",
          flush=True)
med.fast_path_enabled = False
inf, dec, p95 = run("sag")
R["sag_full_path_only"] = {"inference_ms": inf, "decision_ms": dec, "decision_p95_ms": p95}
print(f"sag full path only: decision layer {dec:.3f} ms (p95 {p95:.3f})", flush=True)
json.dump(R, open(OUT, "w"), indent=1)
print("LATENCY COMPLETE", flush=True)
