import json, os, sys, importlib.util
import numpy as np, torch


HERE = os.path.dirname(os.path.abspath(__file__))   # relative paths
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
spec = importlib.util.spec_from_file_location("v1", os.path.join(HERE, "safeagentgrid.py"))
v1 = importlib.util.module_from_spec(spec); sys.argv=["r"]; spec.loader.exec_module(v1)
OUT=os.path.join(HERE, "results", "repair_results.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT,"w"), indent=2); print("[saved]", list(res), flush=True)
def clean(d): return {k:v for k,v in d.items() if isinstance(v,(int,float))}
N, EVAL = 10, 10



med, ags, _ = v1.train_safeagentgrid(30, N, 40, verbose=False, surr_epochs=200)
med.reset_telemetry(); res["reference_repair_clean"] = clean(v1.evaluate(med, ags, N, EVAL)); save()
med.reset_telemetry(); res["reference_repair_fdia"] = clean(v1.evaluate(med, ags, N, EVAL, attack="fdia")); save()
med.reset_telemetry(); res["reference_repair_byz"] = clean(v1.evaluate(med, ags, N, EVAL, attack="byzantine")); save()
med.repair = "median"
med.reset_telemetry(); res["median_repair_fdia"] = clean(v1.evaluate(med, ags, N, EVAL, attack="fdia")); save()
mns = v1.SafeAgentGridMediator(N, 2*N+1, security_enabled=False)
mns.fast_surrogate = med.fast_surrogate; mns.conformal_q = med.conformal_q; mns.surrogate_trained = True
mns.reset_telemetry(); res["no_security_fdia"] = clean(v1.evaluate(mns, ags, N, EVAL, attack="fdia")); save()
for k,v in res.items():


    print(f"{k:28s} viol {v['violations_per_step']:.4f} robust {v['robustness']:.3f} reward {v['avg_reward']:.1f}", flush=True)
print("REPAIR PROBE COMPLETE", flush=True)
