import json, os, sys, importlib.util
import numpy as np, torch




HERE = os.path.dirname(os.path.abspath(__file__))   # relative paths
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
spec = importlib.util.spec_from_file_location("v1", os.path.join(HERE, "safeagentgrid.py"))
v1 = importlib.util.module_from_spec(spec); sys.argv=["s"]; spec.loader.exec_module(v1)
OUT=os.path.join(HERE, "results", "scal40.json")
R=json.load(open(OUT)) if os.path.exists(OUT) else {}
for n in (4,8,10,16):
    if str(n) in R: continue
    med,ags,_=v1.train_safeagentgrid(30,n,40,verbose=False,surr_epochs=200)
    med.reset_telemetry(); r=v1.evaluate(med,ags,n,40)
    ep=np.asarray(r["ep_violations"],float)/200.0
    R[str(n)]={"avg_reward":r["avg_reward"],"per_agent":r["avg_reward"]/n,
               "violations_per_step":r["violations_per_step"],
               "viol_sem":float(ep.std(ddof=1)/np.sqrt(len(ep))),
               "efficiency":r["efficiency"],"latency_ms":r["latency_ms"],
               "alignment":r["alignment"],"fairness_gini":r["fairness_gini"],
               "robustness":r["robustness"],"fast_share":r.get("fast_share_eval",float('nan'))}
    print(f"N={n:2d} rew {r['avg_reward']:10.1f} per-agent {r['avg_reward']/n:9.1f} "
          f"viol {r['violations_per_step']:.4f}+/-{R[str(n)]['viol_sem']:.4f} "
          f"eff {r['efficiency']:.2f} lat {r['latency_ms']:.2f}",flush=True)
    json.dump(R,open(OUT,"w"),indent=2); print("[saved]",sorted(R),flush=True)





print("SCAL40 COMPLETE",flush=True)
