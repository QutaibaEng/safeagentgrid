"""The statistics, LaTeX table rows and figures of the paper.......

PLEASE Reads  the JSON files written by run_final40.py, run_multiseed.py,  
run_chi_sweep.py, run_weight_sensitivity.py, run_conformal_coverage.py, 
run_training_weights.py and run_sweeps40.py (all in ./results) and writes
  results/summary.json            each derived number quoted in the paper
  figures/*.png                   the new / regenerated figures
Usage: python analyze_results.py

"""
import glob
import json
import math
import os

import numpy as np
from scipy import stats



import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
OUT = {}


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None



def wilcoxon_p(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if np.allclose(x, y):
        return float("nan"), float("nan")
    r = stats.wilcoxon(x, y)
    return float(r.statistic), float(r.pvalue)



def ci95(v):
    v = np.asarray(v, float)
    n = len(v)
    m, sd = float(v.mean()), float(v.std(ddof=1)) if n > 1 else 0.0
    h = float(stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)) if n > 1 else 0.0
    return m, sd, m - h, m + h


# ---------------------====-----===--------------------- Table 6 (final40)
F = load("final40.json")
if F:
    t3 = {}
    sag = F["SAG_normal"]
    for key in ("shield", "lag"):
        if f"{key}_normal" not in F:
            continue
        n, f = F[f"{key}_normal"], F.get(f"{key}_fdia")
        row = {k: n[k] for k in ("avg_reward", "violations_per_step", "viol_sem",
                                 "efficiency", "alignment", "fairness_gini",
                                 "latency_ms")}
        if f:
            row["fdia_viol"] = f["violations_per_step"]
            row["fdia_sem"] = f["viol_sem"]
            row["fdia_reward"] = f["avg_reward"]
            row["fdia_robust"] = f["robustness"]
            row["fdia_eff"] = f["efficiency"]
        row["wilcoxon_reward"] = wilcoxon_p(sag["ep_rewards"], n["ep_rewards"])
        row["wilcoxon_viol"] = wilcoxon_p(sag["ep_viol"], n["ep_viol"])
        if f:
            row["wilcoxon_viol_fdia"] = wilcoxon_p(F["SAG_fdia"]["ep_viol"], f["ep_viol"])
        t3[key] = row
    t3["latency_this_machine"] = {"SAG_normal": F["SAG_normal"]["latency_ms"],
                                  "greedy_normal": F["greedy_normal"]["latency_ms"],
                                  "ind_normal": F["ind_normal"]["latency_ms"]}
    
# the consistency check against the reference results of safeagentgrid_results.json
    ORIG = load("../safeagentgrid_results.json")
    if ORIG:
        o = ORIG["final_40ep"]
        mism = []
        for k, v in o.items():
            if k == "training_curve" or k not in F:
                continue
            for m in ("avg_reward", "violations_per_step", "viol_sem", "efficiency",
                      "alignment", "fairness_gini", "robustness"):
                if abs(F[k][m] - v[m]) > 1e-9 * max(1.0, abs(v[m])):
                    mism.append((k, m, v[m], F[k][m]))
        t3["reproduction_mismatches"] = mism
        t3["reproduction_keys_checked"] = sorted(k for k in o if k in F and k != "training_curve")
    OUT["table3"] = t3



# --=============###------------- multi-seed
files = sorted(glob.glob(os.path.join(RES, "multiseed", "seed_*.json")))
if files:
    S = {int(os.path.basename(f)[5:-5]): json.load(open(f)) for f in files}
    seeds = sorted(s for s in S if all(k in S[s] for k in ("SAG_normal", "lag_fdia")))
    methods = [("SAG", "SafeAgentGrid"), ("greedy", "No Mediator"),
               ("proj", "Projection-only"), ("ind", "Independent PG"),
               ("sac", "Single-Agent Const."), ("shield", "Joint safety shield"),
               ("lag", "Lagrangian PG")]


    ms = {"seeds": seeds, "n": len(seeds), "methods": {}}
    for key, label in methods:
        d = {}
        for metric in ("avg_reward", "violations_per_step", "efficiency",
                       "fairness_gini", "alignment", "latency_ms"):
            d[metric] = ci95([S[s][f"{key}_normal"][metric] for s in seeds])
        d["fdia_viol"] = ci95([S[s][f"{key}_fdia"]["violations_per_step"] for s in seeds])
        d["fdia_reward"] = ci95([S[s][f"{key}_fdia"]["avg_reward"] for s in seeds])
        d["per_seed_viol"] = [S[s][f"{key}_normal"]["violations_per_step"] for s in seeds]
        d["per_seed_fdia_viol"] = [S[s][f"{key}_fdia"]["violations_per_step"] for s in seeds]
        d["per_seed_reward"] = [S[s][f"{key}_normal"]["avg_reward"] for s in seeds]
        if key == "SAG":
            d["byz_viol"] = ci95([S[s]["SAG_byz"]["violations_per_step"] for s in seeds])
            d["fast_share"] = ci95([S[s]["SAG_normal"]["fast_share"] for s in seeds])
            d["conformal_q"] = [S[s]["conformal_q"] for s in seeds]
            d["admitted_seeds"] = int(sum(q <= 0.05 for q in d["conformal_q"]))
            d["viol_shortfall"] = ci95([S[s]["SAG_normal"].get("viol_shortfall", np.nan)
                                        for s in seeds])
            d["viol_line"] = ci95([S[s]["SAG_normal"].get("viol_line", np.nan) for s in seeds])
            sl = [S[s]["slsqp_fdia"] for s in seeds if "slsqp_fdia" in S[s]]
            d["slsqp_success"] = float(np.mean([x["success_rate"] for x in sl]))
            d["slsqp_mean_iter"] = float(np.mean([x["mean_iter"] for x in sl]))
            d["slsqp_at_cap"] = float(np.mean([x["at_cap_rate"] for x in sl]))
            curves = np.array([S[s]["training_curve"] for s in seeds])
            d["curve_mean"] = curves.mean(0).tolist()
            d["curve_sd"] = curves.std(0, ddof=1).tolist()
            first, last = curves[:, :5].mean(1), curves[:, -5:].mean(1)
            d["curve_first5"] = ci95(first)
            d["curve_last5"] = ci95(last)
            d["curve_drift_t"] = [float(x) for x in stats.ttest_rel(last, first)]
            d["curve_rel_range"] = float((curves.max(1) - curves.min(1)).mean()
                                         / curves.mean())
        ms["methods"][key] = {"label": label, **d}
    # the paired tests across the seeds: SAG vs each baseline technique .....
    tests = {}
    for key, _ in methods[1:]:
        row = {}
        for metric, better in (("avg_reward", "higher"), ("violations_per_step", "lower")):
            a = np.array([S[s]["SAG_normal"][metric] for s in seeds])
            b = np.array([S[s][f"{key}_normal"][metric] for s in seeds])
            w = wilcoxon_p(a, b)
            tt = stats.ttest_rel(a, b)
            wins = int(np.sum(a > b)) if better == "higher" else int(np.sum(a < b))
            row[metric] = {"wilcoxon": w, "t": [float(tt.statistic), float(tt.pvalue)],
                           "sag_better_seeds": wins, "mean_diff": float(np.mean(a - b))}
        a = np.array([S[s]["SAG_fdia"]["violations_per_step"] for s in seeds])
        b = np.array([S[s][f"{key}_fdia"]["violations_per_step"] for s in seeds])
        row["fdia_viol"] = {"wilcoxon": wilcoxon_p(a, b),
                            "t": [float(x) for x in stats.ttest_rel(a, b)],
                            "sag_better_seeds": int(np.sum(a < b)),
                            "mean_diff": float(np.mean(a - b))}
        a = np.array([S[s]["SAG_normal"]["fairness_gini"] for s in seeds])
        b = np.array([S[s][f"{key}_normal"]["fairness_gini"] for s in seeds])
        row["gini"] = {"wilcoxon": wilcoxon_p(a, b), "sag_better_seeds": int(np.sum(a < b)),
                       "mean_diff": float(np.mean(a - b))}
        tests[key] = row
    ms["tests"] = tests
    OUT["multiseed"] = ms


    # the figure: per-seed violation rates + learning curves....#
    fig, ax = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    order = ["SAG", "shield", "greedy", "ind", "sac", "lag"]
    names = ["SafeAgentGrid", "Joint safety\nshield", "No Mediator", "Independent\nPG MARL",
             "Single-Agent\nConst.", "Lagrangian\nPG MARL"]
    cols = ["tab:blue", "tab:olive", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
    for j, (k, c) in enumerate(zip(order, cols)):
        v = np.array(ms["methods"][k]["per_seed_viol"])
        vf = np.array(ms["methods"][k]["per_seed_fdia_viol"])
        xs = np.full(len(v), j) + np.linspace(-0.12, 0.12, len(v))
        ax[0].scatter(xs - 0.18, v, color=c, s=18, zorder=3,
                      label="normal" if j == 0 else None)
        ax[0].scatter(xs + 0.18, vf, color=c, s=18, marker="^", alpha=0.6, zorder=3,
                      label="20% FDIA" if j == 0 else None)
        ax[0].hlines(v.mean(), j - 0.32, j - 0.04, color="k", lw=1.5)
        ax[0].hlines(vf.mean(), j + 0.04, j + 0.32, color="k", lw=1.5, ls="--")
    ax[0].set_yscale("log")
    ax[0].set_xticks(range(len(order)))
    ax[0].set_xticklabels(names, fontsize=8)
    ax[0].set_ylabel("Violations per step (log scale)")
    ax[0].set_title(f"(a) Per-seed violation rates ({len(seeds)} training seeds)")
    ax[0].grid(alpha=0.3, which="both")
    leg = ax[0].legend(loc="lower right", fontsize=8)
    for h in leg.legend_handles:
        h.set_color("gray")
    m = np.array(ms["methods"]["SAG"]["curve_mean"])
    sd = np.array(ms["methods"]["SAG"]["curve_sd"])
    ep = np.arange(1, len(m) + 1)
    ax[1].plot(ep, m, color="tab:blue", lw=2, label="mean over seeds")
    ax[1].fill_between(ep, m - sd, m + sd, color="tab:blue", alpha=0.2, label="$\\pm$1 SD")
    ax[1].set_xlabel("Training episode")
    ax[1].set_ylabel("Episode reward (training)")
    ax[1].set_title("(b) SafeAgentGrid learning curves")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "multiseed.png"), bbox_inches="tight")
    plt.close(fig)



# -----------#####---------- the chi sweep...
C = load("chi_sweep.json")
if C:
    chis = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = {}
    for c in chis:
        for cond in ("normal", "fdia"):
            k = f"chi{c}_{cond}"
            if k in C:
                rows[k] = {m: C[k][m] for m in ("avg_reward", "violations_per_step",
                                                "viol_sem", "viol_line", "viol_shortfall",
                                                "fairness_gini", "efficiency", "latency_ms")}


    base = C.get("chi0.5_normal")
    if base:
        for c in chis:
            k = f"chi{c}_normal"
            if k in C and c != 0.5:
                rows[k]["wilcoxon_vs_0.5"] = wilcoxon_p(base["ep_viol"], C[k]["ep_viol"])
    OUT["chi"] = {"rows": rows, "analytic": C.get("analytic"),
                  "N16": {k: {m: C[k][m] for m in ("avg_reward", "violations_per_step",
                                                    "viol_sem", "viol_line")}
                          for k in C if k.startswith("N16_")}}
    if all(f"chi{c}_normal" in C for c in chis):
        sig = C["analytic"]["10"]["sigma_line_pu"]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150)
        for cond, col, lab in (("normal", "tab:blue", "Nominal"),
                               ("fdia", "tab:red", "20% FDIA")):
            if not all(f"chi{c}_{cond}" in C for c in chis):
                continue
            v = [C[f"chi{c}_{cond}"]["violations_per_step"] for c in chis]
            e = [1.96 * C[f"chi{c}_{cond}"]["viol_sem"] for c in chis]
            ax[0].errorbar(chis, v, yerr=e, marker="o", color=col, lw=2, capsize=3,
                           label=f"Measured violations/step ({lab})")
        xx = np.linspace(0.3, 1.0, 200)
        ax[0].plot(xx, stats.norm.cdf(-(1 - xx) * 2.5 / sig), "k--", lw=1.5,
                   label="Per-line bound, flow on the tightened boundary (Prop. 4)")
        ax[0].axvline(0.5, color="gray", ls=":", lw=1)
        ax[0].set_yscale("log")
        ax[0].set_xlabel("Tightening fraction $\\chi$")
        ax[0].set_ylabel("Violations per step / probability")
        ax[0].set_title("(a) Safety versus $\\chi$ (full path, $N=10$)")
        ax[0].legend(fontsize=8, loc="upper left")
        ax[0].grid(alpha=0.3, which="both")
        r = [C[f"chi{c}_normal"]["avg_reward"] for c in chis]
        g = [C[f"chi{c}_normal"]["fairness_gini"] for c in chis]
        ax[1].plot(chis, r, "s-", color="tab:green", lw=2, label="Avg. reward (nominal)")
        ax[1].set_xlabel("Tightening fraction $\\chi$")
        ax[1].set_ylabel("Average episode reward", color="tab:green")
        ax[1].tick_params(axis="y", labelcolor="tab:green")
        a2 = ax[1].twinx()
        a2.plot(chis, g, "d--", color="tab:purple", lw=2, label="Gini (nominal)")
        a2.set_ylabel("Fairness (Gini)", color="tab:purple")
        a2.tick_params(axis="y", labelcolor="tab:purple")
        ax[1].axvline(0.5, color="gray", ls=":", lw=1)
        ax[1].set_title("(b) Reward and fairness versus $\\chi$")
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "chi_sweep.png"), bbox_inches="tight")
        plt.close(fig)



# ----------#############+++++===------------------- THE   weights
W = load("weight_sensitivity.json")
if W:
    OUT["weights"] = {k: {m: v[m] for m in ("avg_reward", "violations_per_step", "viol_sem",
                                            "fairness_gini", "efficiency", "robustness")}
                      for k, v in W.items()}
    if "base1.0_normal" in W:
        for k, v in W.items():
            if k.endswith("_normal") and k != "base1.0_normal":
                OUT["weights"][k]["wilcoxon_vs_base"] = wilcoxon_p(
                    W["base1.0_normal"]["ep_viol"], v["ep_viol"])
    if "rho1.0_fdia" in W:
        for k, v in W.items():
            if k.startswith("rho") and k != "rho1.0_fdia":
                OUT["weights"][k]["wilcoxon_vs_base"] = wilcoxon_p(
                    W["rho1.0_fdia"]["ep_viol"], v["ep_viol"])

# ----------===============####------ coverage
CC = load("conformal_coverage.json")
if CC:
    OUT["coverage"] = CC



# ----______##--- the training weights..
TW = load("training_weights.json")
if TW:
    OUT["training_weights"] = {k: {"normal": {m: v["normal"][m] for m in
                                              ("avg_reward", "violations_per_step",
                                               "viol_sem", "fairness_gini")},
                                   "fdia": {m: v["fdia"][m] for m in
                                            ("violations_per_step", "viol_sem")},
                                   "q": v["conformal_q"]} for k, v in TW.items()}




######## -------- THE sweeps at 40 episodes
S40 = load("sweeps40.json")
if S40:
    OUT["sweeps40"] = {k: {m: v[m] for m in ("avg_reward", "violations_per_step", "viol_sem",
                                             "robustness", "detect_rate")}
                       for k, v in S40.items()}
    if "repair_reference_fdia" in S40 and "repair_median_fdia" in S40:
        OUT["sweeps40"]["repair_wilcoxon"] = wilcoxon_p(
            S40["repair_reference_fdia"]["ep_viol"], S40["repair_median_fdia"]["ep_viol"])
    keys = [("stealth_1_2", "U[1,2]"), ("stealth_2_5", "U[2,5]"),
            ("stealth_5_10", "U[5,10]"), ("stealth_20_35", "U[20,35]")]
    if all(k in S40 for k, _ in keys):
        det = [S40[k]["detect_rate"] for k, _ in keys]
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        bars = ax.bar([l for _, l in keys], det,
                      color=["tab:red" if d < 99.95 else "tab:green" for d in det])
        ax.axhline(100, ls=":", color="gray")
        for b, d in zip(bars, det):
            ax.text(b.get_x() + b.get_width() / 2, d + 1.0,
                    f"{d:.1f}%" if d < 99.95 else f"{d:.0f}%", ha="center", fontsize=10)
        ax.set_ylim(0, 112)
        ax.set_ylabel("Anomaly detection rate (%)")
        ax.set_xlabel("FDIA perturbation magnitude $\\delta$ (action units; action range $\\pm$20)")
        fig.suptitle("Detector envelope: MAD detection degrades for bounded perturbations",
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "stealth_detection.png"), bbox_inches="tight")
        plt.close(fig)
    etas = [0.05, 0.5, 2.0, 10.0]
    if all(f"eta{e:g}_{t}" in S40 for e in etas for t in ("secured", "unsecured")):
        fig, ax1 = plt.subplots(figsize=(8.5, 5.2), dpi=150)
        von = [S40[f"eta{e:g}_secured"]["violations_per_step"] for e in etas]
        voff = [S40[f"eta{e:g}_unsecured"]["violations_per_step"] for e in etas]
        ron = [S40[f"eta{e:g}_secured"]["avg_reward"] for e in etas]
        roff = [S40[f"eta{e:g}_unsecured"]["avg_reward"] for e in etas]
        ax1.plot(etas, von, "o-", color="tab:blue", lw=2, label="Violations, security ON")
        ax1.plot(etas, voff, "o--", color="tab:red", lw=2, label="Violations, security OFF")
        ax1.set_xscale("log")
        ax1.set_xlabel("Consensus weight $\\eta$ (delegated agent authority)")
        ax1.set_ylabel("Violations per step")
        ax1.grid(alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(etas, ron, "s-", color="tab:green", alpha=0.8, label="Reward, security ON")
        ax2.plot(etas, roff, "s--", color="tab:olive", alpha=0.8, label="Reward, security OFF")
        ax2.set_ylabel("Average episode reward")
        ax1.axvspan(0.3, 0.9, color="gold", alpha=0.18)
        ax1.text(0.52, 0.4, "recommended\noperating point", ha="center", fontsize=9,
                 transform=ax1.get_xaxis_transform())
        l1, b1 = ax1.get_legend_handles_labels()
        l2, b2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, b1 + b2, fontsize=8, loc="upper left")
        fig.suptitle("Delegation dial: the security module's value is governed by $\\eta$ "
                     "(20% FDIA)", fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "eta_sweep.png"), bbox_inches="tight")
        plt.close(fig)



# -###-----------THE comparison bar (Figure) ....
if F and all(f"{k}_normal" in F for k in ("shield", "lag")):
    rows = [("SafeAgentGrid\n(Ours)", "SAG"), ("No Mediator", "greedy"),
            ("Projection-only\nsafety layer", "proj"), ("Independent PG\nMARL (CTDE)", "ind"),
            ("Single-Agent\nConstitutional", "sac"), ("Joint safety\nshield", "shield"),
            ("Lagrangian\nPG MARL", "lag")]
    eff = [F[f"{k}_normal"]["efficiency"] for _, k in rows]
    ali = [100 * F[f"{k}_normal"]["alignment"] for _, k in rows]
    saf = [100 * (1 - F[f"{k}_normal"]["violations_per_step"]) for _, k in rows]
    rew = [F[f"{k}_normal"]["avg_reward"] for _, k in rows]
    x = np.arange(len(rows)); w = 0.2
    fig, ax1 = plt.subplots(figsize=(14, 6.5), dpi=150)
    b1 = ax1.bar(x - 1.5 * w, eff, w, label="Efficiency (%)", color="tab:green")
    b2 = ax1.bar(x - 0.5 * w, ali, w, label="Alignment (%)", color="orange")
    b3 = ax1.bar(x + 0.5 * w, saf, w, label="Safety Score (%)", color="tab:red")
    ax1.set_ylabel("Percentage / Scaled Score")
    ax1.set_ylim(0, 118)
    ax2 = ax1.twinx()
    b4 = ax2.bar(x + 1.5 * w, rew, w, label="Avg. Reward", color="tab:blue")
    ax2.set_ylabel("Avg. Reward")
    ax2.set_ylim(0, 250000)
    for bars, vals, fmt, axx in ((b1, eff, "{:.1f}", ax1), (b2, ali, "{:.1f}", ax1),
                                 (b3, saf, "{:.1f}", ax1), (b4, rew, "{:.0f}", ax2)):
        for b, v in zip(bars, vals):
            axx.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.005 +
                     (0.6 if axx is ax1 else 1500), fmt.format(v), ha="center",
                     fontsize=6.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels([n for n, _ in rows], fontsize=8)
    l1, bb1 = ax1.get_legend_handles_labels(); l2, bb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, bb1 + bb2, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle("SafeAgentGrid vs. baselines implemented in the same environment "
                 "(N=10, 40 evaluation episodes)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "comparison_bar_improved.png"), bbox_inches="tight")
    plt.close(fig)


# -###------------------- THE robustness in (run_advantage*.py)
ADV = {}
for part in ("A", "B"):
    d = load(f"advantage_{part}.json")
    if d:
        ADV.update(d)
if ADV:
    CONDS = [("normal", "Nominal"), ("fdia", "FDIA, gap-widening U[20,35]"),
             ("flow_20_35", "FDIA, network-targeted U[20,35]"),
             ("flow_2_5", "FDIA, network-targeted U[2,5]"),
             ("flow_1_2", "FDIA, network-targeted U[1,2]"),
             ("stealth12", "FDIA, gap-widening U[1,2] (stealth)"),
             ("byz", "Byzantine, floor(N/3) agents")]
    hh = {"conditions": [c for c, _ in CONDS], "labels": dict(CONDS), "rows": {}}
    for c, _ in CONDS:
        row = {}
        for m in ("SAG", "SAGnosec", "shield"):
            k = f"{m}_{c}"
            if k in ADV:
                x = ADV[k]
                row[m] = {q: x[q] for q in ("violations_per_step", "viol_sem", "avg_reward",
                                             "fairness_gini")}
        for m in ("SAGnosec", "shield"):
            if f"SAG_{c}" in ADV and f"{m}_{c}" in ADV:
                row[f"wilcoxon_SAG_vs_{m}"] = wilcoxon_p(ADV[f"SAG_{c}"]["ep_viol"],
                                                        ADV[f"{m}_{c}"]["ep_viol"])
                row[f"wilcoxon_reward_SAG_vs_{m}"] = wilcoxon_p(ADV[f"SAG_{c}"]["ep_rewards"],
                                                               ADV[f"{m}_{c}"]["ep_rewards"])
        hh["rows"][c] = row
    attacked = [c for c, _ in CONDS if c != "normal"]
    for m in ("SAG", "SAGnosec", "shield"):
        vals = [hh["rows"][c][m]["violations_per_step"] for c in attacked
                if m in hh["rows"][c]]
        if len(vals) == len(attacked):
            hh[f"worst_attacked_{m}"] = max(vals)
            hh[f"mean_attacked_{m}"] = float(np.mean(vals))
    # ten-seed confirmation (run_advantage_ms.py)
    msd = os.path.join(RES, "advantage_ms")
    if os.path.isdir(msd):
        SG = {int(f[9:-5]): json.load(open(os.path.join(msd, f)))
              for f in os.listdir(msd) if f.startswith("SAG_seed_")}
        SH = {int(f[12:-5]): json.load(open(os.path.join(msd, f)))
              for f in os.listdir(msd) if f.startswith("shield_seed_")}
        # seed 42 of SafeAgentGrid is evaluated by run_advantage.py (part A)
        if "SAG_flow_20_35" in ADV and 42 not in SG:
            SG[42] = {"SAG_normal": ADV["SAG_normal"], "SAG_fdia": ADV["SAG_fdia"],
                      "SAG_byz": ADV["SAG_byz"], "SAG_flow": ADV["SAG_flow_20_35"]}
        seeds = sorted(s_ for s_ in SG if s_ in SH and "SAG_flow" in SG[s_]
                       and "shield_flow" in SH[s_])
        ms10 = {"seeds": seeds, "n": len(seeds)}
 


       # The reproduction check against results/multiseed.............
        repro = []
        for s_ in seeds:
            f0 = os.path.join(RES, "multiseed", f"seed_{s_}.json")
            if os.path.exists(f0):
                old = json.load(open(f0))
                for k in ("SAG_normal", "SAG_fdia", "SAG_byz"):
                    if k in SG[s_] and k in old:
                        repro.append(abs(SG[s_][k]["violations_per_step"]
                                         - old[k]["violations_per_step"]) < 1e-12)
                for c in ("normal", "fdia"):
                    k = f"shield_{c}"
                    if k in SH[s_] and k in old:
                        repro.append(abs(SH[s_][k]["violations_per_step"]
                                         - old[k]["violations_per_step"]) < 1e-12)
        ms10["reproduced_multiseed_values"] = [int(sum(repro)), len(repro)]
        for c in ("normal", "fdia", "byz", "flow"):
            a = np.array([SG[s_][f"SAG_{c}"]["violations_per_step"] for s_ in seeds])
            b = np.array([SH[s_][f"shield_{c}"]["violations_per_step"] for s_ in seeds])
            ra = np.array([SG[s_][f"SAG_{c}"]["avg_reward"] for s_ in seeds])
            rb = np.array([SH[s_][f"shield_{c}"]["avg_reward"] for s_ in seeds])
            ga = np.array([SG[s_][f"SAG_{c}"]["fairness_gini"] for s_ in seeds])
            gb = np.array([SH[s_][f"shield_{c}"]["fairness_gini"] for s_ in seeds])
            ms10[c] = {"SAG": ci95(a), "shield": ci95(b),
                       "SAG_reward": ci95(ra), "shield_reward": ci95(rb),
                       "SAG_gini": ci95(ga), "shield_gini": ci95(gb),
                       "per_seed_SAG": a.tolist(), "per_seed_shield": b.tolist(),
                       "wilcoxon": wilcoxon_p(a, b), "sag_better_seeds": int(np.sum(a < b)),
                       "wilcoxon_reward": wilcoxon_p(ra, rb),
                       "sag_better_reward_seeds": int(np.sum(ra > rb)),
                       "sag_better_gini_seeds": int(np.sum(ga < gb))}
        hh["tenseed"] = ms10
    OUT["robustness"] = hh




def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o




json.dump(_clean(OUT), open(os.path.join(RES, "summary.json"), "w"), indent=1)
print("wrote", os.path.join(RES, "summary.json"), "sections:", list(OUT))
