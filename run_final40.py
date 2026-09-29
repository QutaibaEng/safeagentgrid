"""NOW Regenerate EVERY table at a uniform 40 evaluation episodes so that all of the 
tables report the same quantity identically.    Resumable: saves after each stage."""




import json, os, sys, importlib.util
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))   # relative paths
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
spec = importlib.util.spec_from_file_location("v1", os.path.join(HERE, "safeagentgrid.py"))
v1 = importlib.util.module_from_spec(spec); sys.argv=["f"]; spec.loader.exec_module(v1)
OUT=os.path.join(HERE, "results", "final40.json")
R = json.load(open(OUT)) if os.path.exists(OUT) else {}
def save(): json.dump(R, open(OUT,"w"), indent=2); print("[saved]", sorted(R), flush=True)
def pack(r):
    ep=np.asarray(r["ep_violations"],float)/200.0
    return {"avg_reward":r["avg_reward"],"violations_per_step":r["violations_per_step"],
            "viol_sem":float(ep.std(ddof=1)/np.sqrt(len(ep))),
            "efficiency":r["efficiency"],"alignment":r["alignment"],
            "fairness_gini":r["fairness_gini"],"latency_ms":r["latency_ms"],
            "robustness":r["robustness"],
            "fast_share":r.get("fast_share_eval",float("nan")),
            "ep_rewards":r["ep_rewards"],"ep_viol":ep.tolist()}
N,E=10,40
med,ags,curve=v1.train_safeagentgrid(30,N,40,verbose=False,surr_epochs=200)
R["training_curve"]=curve
print("model ready",flush=True)




def stage(key, fn):
    if key in R: return
    R[key]=pack(fn()); 
    print(f"{key:34s} rew {R[key]['avg_reward']:9.1f} viol {R[key]['violations_per_step']:.4f}"
          f" +/-{R[key]['viol_sem']:.4f} eff {R[key]['efficiency']:.2f} lat {R[key]['latency_ms']:.2f}",flush=True)
    save()



def ev(**kw):
    med.reset_telemetry(); return v1.evaluate(med,ags,N,E,**kw)

stage("SAG_normal", lambda: ev())
stage("SAG_fdia",   lambda: ev(attack="fdia"))
stage("SAG_byz",    lambda: ev(attack="byzantine"))
stage("greedy_normal", lambda: v1.evaluate_policy_only(ags,N,E,mode="greedy"))
stage("greedy_fdia",   lambda: v1.evaluate_policy_only(ags,N,E,mode="greedy",attack="fdia"))
stage("proj_normal",   lambda: v1.evaluate_policy_only(ags,N,E,mode="projection"))
stage("proj_fdia",     lambda: v1.evaluate_policy_only(ags,N,E,mode="projection",attack="fdia"))

if "ind_normal" not in R:
    ag_ind=v1.train_unmediated(30,N,shared_policy=False,constitutional=False)
    torch.save([a.state_dict() for a in ag_ind],os.path.join(HERE, "results", "ag_ind.pt"))
    stage("ind_normal", lambda: v1.evaluate_policy_only(ag_ind,N,E,mode="greedy"))
    stage("ind_fdia",   lambda: v1.evaluate_policy_only(ag_ind,N,E,mode="greedy",attack="fdia"))
if "sac_normal" not in R:
    ag_sac=v1.train_unmediated(30,N,shared_policy=True,constitutional=True)
    stage("sac_normal", lambda: v1.evaluate_policy_only(ag_sac,N,E,mode="projection"))
    stage("sac_fdia",   lambda: v1.evaluate_policy_only(ag_sac,N,E,mode="projection",attack="fdia"))

# ---- ablation ============..........



def clone(**kw):
    m=v1.SafeAgentGridMediator(N,2*N+1,**kw)
    m.fast_surrogate=med.fast_surrogate; m.conformal_q=med.conformal_q; m.surrogate_trained=True
    return m
if "abl_nofast_normal" not in R:
    m=clone(fast_path_enabled=False)
    stage("abl_nofast_normal", lambda: (m.reset_telemetry(), v1.evaluate(m,ags,N,E))[1])
    stage("abl_nofast_fdia",   lambda: (m.reset_telemetry(), v1.evaluate(m,ags,N,E,attack="fdia"))[1])
if "abl_nosec_normal" not in R:
    m2=clone(security_enabled=False)
    stage("abl_nosec_normal", lambda: (m2.reset_telemetry(), v1.evaluate(m2,ags,N,E))[1])
    stage("abl_nosec_fdia",   lambda: (m2.reset_telemetry(), v1.evaluate(m2,ags,N,E,attack="fdia"))[1])
if "abl_nogat_normal" not in R:
    mg,ag_g,_=v1.train_safeagentgrid(30,N,40,use_gat=False,verbose=False,surr_epochs=200)
    stage("abl_nogat_normal", lambda: (mg.reset_telemetry(), v1.evaluate(mg,ag_g,N,E))[1])
    stage("abl_nogat_fdia",   lambda: (mg.reset_telemetry(), v1.evaluate(mg,ag_g,N,E,attack="fdia"))[1])


# ---- baseline techniques that coordinate the agents at the network level: the joint safety
# shield and the Lagrangian constrained PG MARL method.
if "shield_normal" not in R:          # joint safety shield (textbook joint safety layer)
    ag_sh = v1.train_unmediated(30, N, shared_policy=False, constitutional=False,
                                shield=v1.JointSafetyShield())
    stage("shield_normal", lambda: v1.evaluate_policy_only(
        ag_sh, N, E, mode="shield", shield=v1.JointSafetyShield()))
    stage("shield_fdia", lambda: v1.evaluate_policy_only(
        ag_sh, N, E, mode="shield", attack="fdia", shield=v1.JointSafetyShield()))
if "lag_normal" not in R:
    ag_lag = v1.train_unmediated(30, N, shared_policy=False, constitutional=False,
                                 lagrangian=True, cost_limit=0.0, lag_lr=50.0)
    stage("lag_normal", lambda: v1.evaluate_policy_only(ag_lag, N, E, mode="greedy"))
    stage("lag_fdia",   lambda: v1.evaluate_policy_only(ag_lag, N, E, mode="greedy",
                                                        attack="fdia"))
print("FINAL40 COMPLETE",flush=True)
