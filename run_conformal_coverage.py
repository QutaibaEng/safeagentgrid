"""The exchangeability of the split-conformal certificate and what
happens when it is violated.....


On the reference model (training seed 42) we record, on EVERY decision step,
the alignment that the filtered fast-path (surrogate) action WOULD have had
(`shadow_log`), whether or not the fast path was actually taken.  The empirical
coverage  P[ alpha_shadow >= 1 - q ]  is then compared with the nominal level
1 - delta = 0.90 under
  (a) the deployment distribution (nominal operation),
  (b) three attack-induced shifts: 20% FDIA, Byzantine, stealth FDIA U[1,2],
  (c) a benign exogenous shift: the diurnal demand swing doubled (+/-40%).
The executed actions are NOT really affected by the shadow computation.
We also report the lag-1 autocorrelation of the calibration scores (a
diagnostic of the serial dependence that makes exchangeability approximate).



Usage: python run_conformal_coverage.py     (results/conformal_coverage.json)
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
sys.argv = ["cc"]
spec.loader.exec_module(v1)

OUT = os.path.join(HERE, "results", "conformal_coverage.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
R = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(R, open(OUT, "w"), indent=1)




N, E = 10, 40
med, ags, _ = v1.train_safeagentgrid(30, N, 40, verbose=False, surr_epochs=200, seed=42)
q = float(med.conformal_q)
R["conformal_q_deployed"] = q
R["delta"] = med.conformal_delta
save()

_orig_demand = v1.SmartGridEnv._demand_at


def _demand_doubled_swing(self, t):
    return (100.0 * self.N + 40.0 * self.N * np.sin(2 * np.pi * t / 48)
            + np.random.normal(0, 0.8 * self.N))



CONDS = [("nominal", dict()),
         ("fdia20", dict(attack="fdia")),
         ("byzantine", dict(attack="byzantine")),
         ("stealth_U1_2", dict(attack="fdia", fdia_lo=1.0, fdia_hi=2.0)),
         ("demand_swing_x2", dict()),
         # detector-only gating: no high-risk alert, fast path gated by the
         # runtime gates alone (what happens if the attack is not announced)
         ("fdia20_detector_only", dict(attack="fdia", oracle_high_risk=False)),
         ("byzantine_detector_only", dict(attack="byzantine", oracle_high_risk=False)),
         ("stealth_U1_2_detector_only", dict(attack="fdia", fdia_lo=1.0, fdia_hi=2.0,
                                             oracle_high_risk=False))]

for name, kw in CONDS:
    if name in R:
        continue
    v1.SmartGridEnv._demand_at = (_demand_doubled_swing if name == "demand_swing_x2"
                                  else _orig_demand)
    med.shadow_log = []
    med.reset_telemetry()
    t0 = time.time()
    r = v1.evaluate(med, ags, N, E, **kw)
    sh = np.asarray(med.shadow_log, dtype=float)       # (alpha, fast, flagged)
    alpha, fast, flag = sh[:, 0], sh[:, 1].astype(bool), sh[:, 2].astype(bool)
    cov = lambda m: float(np.mean(alpha[m] >= 1.0 - q - 1e-12)) if m.any() else float("nan")
    cov95 = lambda m: float(np.mean(alpha[m] >= 0.95 - 1e-12)) if m.any() else float("nan")
    allm = np.ones_like(fast, dtype=bool)
    R[name] = {"steps": int(len(alpha)),
               "coverage_all_steps": cov(allm),
               "coverage_admitted_steps": cov(fast),
               "coverage_unflagged_steps": cov(~flag),
               "coverage_flagged_steps": cov(flag),
               "coverage95_all_steps": cov95(allm),
               "coverage95_admitted_steps": cov95(fast),
               "fast_share": float(fast.mean()),
               "flag_share": float(flag.mean()),
               "mean_shadow_alignment": float(alpha.mean()),
               "violations_per_step": r["violations_per_step"],
               "avg_reward": r["avg_reward"]}
    med.shadow_log = None
    print(f"{name:16s} cov(all) {R[name]['coverage_all_steps']:.3f} "
          f"cov(admitted) {R[name]['coverage_admitted_steps']:.3f} "
          f"fast {R[name]['fast_share']:.3f} flagged {R[name]['flag_share']:.3f} "
          f"viol {r['violations_per_step']:.4f} ({time.time()-t0:.0f}s)", flush=True)
    save()
v1.SmartGridEnv._demand_at = _orig_demand


# --__== serial dependence of the calibration scores (same code path as
# ###########calibrate_conformal, on a fresh trajectory; diagnostic only)
if "calibration_diagnostics" not in R:
    np.random.seed(123)
    env = v1.SmartGridEnv(N=N)
    scores = []
    s = env.reset()
    with torch.no_grad():
        for _ in range(150):
            props = med._sample_proposals(ags, s, N)
            feat = np.concatenate([s, props]).astype(np.float32)
            a = med.constitution.filter(med.fast_surrogate(feat).cpu().numpy())
            scores.append(1.0 - med.alignment_score(a, env))
            env.step(a)
            s = env._get_state()
    x = np.asarray(scores)
    lag1 = float(np.corrcoef(x[:-1], x[1:])[0, 1]) if x.std() > 0 else 0.0
    R["calibration_diagnostics"] = {"m": 150, "mean_score": float(x.mean()),
                                    "frac_zero": float(np.mean(x == 0)),
                                    "lag1_autocorr": lag1}
    print("calibration diagnostics", R["calibration_diagnostics"], flush=True)
    save()
print("CONFORMAL COVERAGE COMPLETE", flush=True)
