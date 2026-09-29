"""THE decisive experiment: sweep the consensus weight eta, which will set how much
authority the mediator will delegate to the agents' proposals......

At low eta the mediator ignores the agents entirely --     so the corrupted proposals
cannot hurt, and the security module has nothing to do with.  At high eta the agents
drive the executed action --     so the FDIA becomes damaging and the security module
should start paying for itself.  Note that the  under attack every step takes the full path,
so eta can be swept at the evaluation time without retraining the surrogate.....
"""



import json, os, sys, importlib.util
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))   # relative paths
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
spec = importlib.util.spec_from_file_location("v1", os.path.join(HERE, "safeagentgrid.py"))
v1 = importlib.util.module_from_spec(spec); sys.argv = ["e"]; spec.loader.exec_module(v1)
OUT = os.path.join(HERE, "results", "eta_results.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(res, open(OUT, "w"), indent=2); print("[saved]", len(res), flush=True)
def clean(d): return {k: v for k, v in d.items() if isinstance(v, (int, float))}

N, EVAL = 10, 10
med, ags, _ = v1.train_safeagentgrid(30, N, 40, verbose=False, surr_epochs=200)
mns = v1.SafeAgentGridMediator(N, 2 * N + 1, security_enabled=False)
mns.fast_surrogate = med.fast_surrogate; mns.conformal_q = med.conformal_q
mns.surrogate_trained = True

for eta in (0.05, 0.5, 2.0, 10.0):
    for tag, m in (("secured", med), ("unsecured", mns)):
        key = f"eta{eta}_{tag}"
        if key in res:
            continue
        m.ETA = eta
        m.reset_telemetry()
        r = clean(v1.evaluate(m, ags, N, EVAL, attack="fdia"))
        res[key] = r
        print(f"eta={eta:5.2f} {tag:9s} | viol {r['violations_per_step']:.4f} | "
              f"robust {r['robustness']:.3f} | reward {r['avg_reward']:9.1f}", flush=True)
        save()


print("ETA SWEEP COMPLETE", flush=True)
