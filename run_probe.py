"""The   targeted follow-up: is the security module's poor showing under FDIA caused
by the 'tighter_action_box' constitutional principle removing exactly the
control authority the mediator needs? NOW    Re-runs the defended/undefended
comparison with that principle removed from the library."""



import json, os, sys, importlib.util
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))   # relative paths
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
spec = importlib.util.spec_from_file_location("v1", os.path.join(HERE, "safeagentgrid.py"))
v1 = importlib.util.module_from_spec(spec); sys.argv = ["p"]; spec.loader.exec_module(v1)





OUT = os.path.join(HERE, "results", "probe_results.json")
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save():
    json.dump(res, open(OUT, "w"), indent=2); print("[saved]", list(res), flush=True)
def clean(d): return {k: v for k, v in d.items() if isinstance(v, (int, float))}

N, EVAL = 10, 10
mediator, agents, _ = v1.train_safeagentgrid(30, N, 40, verbose=False, surr_epochs=200)

# (a) as-shipped
mediator.reset_telemetry()
res["with_box_tightening_fdia"] = clean(v1.evaluate(mediator, agents, N, EVAL, attack="fdia"))
save()

# (b) library without the action-box tightening
v1.Constitution.LIBRARY = {
    "higher_reserve_margin": dict(reserve=0.08),
    "stronger_fairness": dict(nu=0.6),
}
mediator.constitution = v1.Constitution()
mediator.reset_telemetry()
res["no_box_tightening_fdia"] = clean(v1.evaluate(mediator, agents, N, EVAL, attack="fdia"))
save()

# --------(c) security disabled entirely, same library
med_ns = v1.SafeAgentGridMediator(N, 2*N+1, security_enabled=False)
med_ns.fast_surrogate = mediator.fast_surrogate
med_ns.conformal_q = mediator.conformal_q
med_ns.surrogate_trained = True
med_ns.reset_telemetry()
res["no_security_fdia"] = clean(v1.evaluate(med_ns, agents, N, EVAL, attack="fdia"))
save()




for k, v in res.items():
    print(f"{k:32s} viol {v['violations_per_step']:.4f} robust {v['robustness']:.3f} "
          f"reward {v['avg_reward']:.1f}", flush=True)
print("PROBE COMPLETE", flush=True)
