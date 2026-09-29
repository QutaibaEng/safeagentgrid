"""Runs the pieces lost when the main process has been   killed: the matched ablation
and the scalability sweep.  the results have been  written to disk after EVERY stage so a
kill never costs more than one stage.    ...  

Further, the seeds are set at module import exactly   as in the main pipeline, and the
train_safeagentgrid is the first call, so the retrained N=10 model is identical
to the one utilized for the main results......



"""
import json, os, sys, importlib.util
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))   # relative paths
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
spec = importlib.util.spec_from_file_location("v1", os.path.join(HERE, "safeagentgrid.py"))
v1 = importlib.util.module_from_spec(spec)
sys.argv = ["rest"]
spec.loader.exec_module(v1)

OUT = os.path.join(HERE, "results", "rest_results.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] stages: {list(res.keys())}", flush=True)


def clean(d):
    return {k: v for k, v in d.items() if isinstance(v, (int, float))}


N, EVAL, TRAIN_EPS, SURR, SURR_EP = 10, 10, 30, 40, 200
train_kwargs = dict(episodes=TRAIN_EPS, N=N, surrogate_collections=SURR,
                    surr_epochs=SURR_EP)

print("=== retraining the reference N=10 model (deterministic) ===", flush=True)
mediator, agents, curve = v1.train_safeagentgrid(
    TRAIN_EPS, N, SURR, verbose=True, surr_epochs=SURR_EP)
mediator.reset_telemetry()
norm = v1.evaluate(mediator, agents, N, EVAL)
print(f"reference check -> reward {norm['avg_reward']:.1f} "
      f"viol {norm['violations_per_step']:.4f} lat {norm['latency_ms']:.2f} "
      f"(main run gave 200258.4 / 0.0120 / 2.88)", flush=True)
res["reference_recheck"] = clean(norm)
res["training_curve"] = curve
save()

torch.save({"surrogate": mediator.fast_surrogate.state_dict()},
           os.path.join(HERE, "results", "ckpt_surrogate.pt"))




# ---------------- ablation -------============--###---
if "ablation" not in res:
    ab = v1.run_ablation(mediator, agents, train_kwargs, N, EVAL)
    res["ablation"] = {k: {m: clean(v) for m, v in r.items()}
                       for k, r in ab.items()}
    save()

######----============------ scalability (one N at a time) ----------------

res.setdefault("scalability", {})
for n in (4, 8, 16):
    if str(n) in res["scalability"]:
        continue
    print(f"\n=== scalability: training N={n} ===", flush=True)
    med, ags, _ = v1.train_safeagentgrid(TRAIN_EPS, n, SURR, verbose=False,
                                         surr_epochs=SURR_EP)
    med.reset_telemetry()
    r = v1.evaluate(med, ags, n, EVAL)
    print(f"N={n:2d} | Reward {r['avg_reward']:10.1f} | per-agent "
          f"{r['avg_reward']/n:9.1f} | Viol {r['violations_per_step']:.4f} | "
          f"Eff {r['efficiency']:.2f}% | Lat {r['latency_ms']:.2f} ms", flush=True)
    res["scalability"][str(n)] = clean(r)
    save()

res["scalability"]["10"] = clean(norm)
save()
print("\nALL STAGES COMPLETE", flush=True)
