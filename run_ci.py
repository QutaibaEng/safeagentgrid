"""    the higher-episode re-evaluation of the two sweeps whose curves are non-monotone,
with the per-episode violation counts retained so that the confidence intervals can be
reported.  The point is NOT to change the measurements but to establish whether
the observed wiggles are real. the resumable: saves after every evaluation.  .....

"""



import json, os, sys, importlib.util
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))   # relative paths........
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
spec = importlib.util.spec_from_file_location("v1", os.path.join(HERE, "safeagentgrid.py"))
v1 = importlib.util.module_from_spec(spec); sys.argv = ["ci"]; spec.loader.exec_module(v1)

OUT = os.path.join(HERE, "results", "ci_results.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT, "w"), indent=2); print("[saved]", len(res), flush=True)

N, EVAL = 10, 40
med, ags, _ = v1.train_safeagentgrid(30, N, 40, verbose=False, surr_epochs=200)
print("model ready", flush=True)

def record(key, r):
    ep = np.asarray(r["ep_violations"], dtype=float) / 200.0
    res[key] = {"violations_per_step": r["violations_per_step"],
                "ep_viol_per_step": ep.tolist(),
                "sem": float(ep.std(ddof=1) / np.sqrt(len(ep))),
                "robustness": r["robustness"], "avg_reward": r["avg_reward"],
                "latency_ms": r["latency_ms"],
                "fast_share": r.get("fast_share_eval", float("nan")),
                "fairness_gini": r["fairness_gini"]}
    print(f"{key:22s} viol {r['violations_per_step']:.4f} "
          f"+/- {res[key]['sem']:.4f} (SEM over {EVAL} episodes)", flush=True)
    save()

for f in (0.0, 0.10, 0.20, 0.30, 0.40):
    k = f"attack_{f:.2f}"
    if k in res: continue
    med.reset_telemetry()
    record(k, v1.evaluate(med, ags, N, EVAL,
                          attack=None if f == 0.0 else "fdia", attack_frac=f))

for g in (0.0, 0.90, 0.95, 0.99, 2.0):
    k = f"gate_{g:.2f}"
    if k in res: continue
    med.fast_gate = g; med.reset_telemetry()
    record(k, v1.evaluate(med, ags, N, EVAL))
med.fast_gate = 0.99


print("CI RUN COMPLETE", flush=True)
