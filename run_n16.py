import json, os, sys, importlib.util
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))   # relative paths
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
spec = importlib.util.spec_from_file_location("v1", os.path.join(HERE, "safeagentgrid.py"))

v1 = importlib.util.module_from_spec(spec); sys.argv = ["n"]; spec.loader.exec_module(v1)
OUT = os.path.join(HERE, "results", "rest_results.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
med, ags, _ = v1.train_safeagentgrid(30, 16, 40, verbose=False, surr_epochs=200)
med.reset_telemetry()


r = v1.evaluate(med, ags, 16, 10)
r = {k: v for k, v in r.items() if isinstance(v, (int, float))}
print(f"N=16 | Reward {r['avg_reward']:10.1f} | per-agent {r['avg_reward']/16:9.1f} | "
      f"Viol {r['violations_per_step']:.4f} | Eff {r['efficiency']:.2f}% | "
      f"Lat {r['latency_ms']:.2f} ms", flush=True)


res.setdefault("scalability", {})["16"] = r
json.dump(res, open(OUT, "w"), indent=2)
print("N16 COMPLETE", flush=True)
