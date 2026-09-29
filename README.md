# THE SafeAgentGrid Technique ...... 
# For help or support, please contact Qutaiba Alasad at qutaibaeng@tu.edu.iq

The reference implementation of *SafeAgentGrid: A Hierarchical Constitutional Multi-Agent
Framework for Trustworthy Autonomous Resource Optimization in Smart Energy
Infrastructures*..................

This repository contains the complete framework   (graph-augmented constitutional agents,
the hybrid neural-surrogate mediator with a conformally gated fast path, the security
module, and the 8-bit quantized deployment path),   the DC power-flow simulation
environment, every baseline used in the paper, and the scripts that reproduce every
table and figure, together with the raw per-episode results. ...



## THe main requirements............

```
python >= 3.10
torch, scipy, numpy, matplotlib
```

## For quick start

```bash
python safeagentgrid.py --medium     # full pipeline (30 training / 40 evaluation episodes)
```



## Reproducing the paper...

All of the  scripts utilized the paths relative to this folder, write their raw results to `results/`,
save after each stage, and resume if they are interrupted...........

```bash
python run_final40.py               # Tables 6, 9, 12 (main comparison, ablation, adversarial)
python run_scal40.py                  # Table 10 (scalability, N = 4, 8, 10, 16)
python run_ci.py                   # Tables 11, 13 (attack severity and gate sweeps)
python run_multiseed.py --seeds 42 43 44 45 46 47 48 49 50 51
                                  # Table 7, Figure 2 (ten independent training seeds)
python run_advantage.py --part A     # Table 8 (SafeAgentGrid, with and without the security module)
python run_advantage.py --part B     # Table 8 (joint safety shield)
python run_advantage_ms.py --method SAG        # Section 4.4, ten training seeds
python run_advantage_ms.py --method shield
python run_chi_sweep.py              # Table 17, Figure 3 (tightening fraction chi, and N = 16)
python run_weight_sensitivity.py     # Table 18 (objective weights lambda, mu, nu, rho_t)
python run_training_weights.py       # Table 19 (training-reward weights beta, lambda_sec)
python run_conformal_coverage.py     # Table 15 (conformal coverage under distribution shift)
python run_sweeps40.py               # Tables 14, 16 (stealth, delegation and repair sweeps)
python run_latency.py                # like-for-like latency benchmark
python analyze_results.py            # statistics, figures and results/summary.json
```



Training the seed 42 is the reference model of the paper, and seed 42 of `run_multiseed.py`
is bit-identical to it. The pipeline is deterministic given the seeds: a from-scratch run
reproduces every reward and violation value of the paper exactly (`analyze_results.py`
checks `run_final40.py` against `safeagentgrid_results.json`: 17 result sets, 0
mismatches). Latency is the one quantity that does not reproduce across machines, and has been
reported in the paper as a mean over seeds for that reason....

## The data provenance

The demand and supply processes are **synthetic parametric processes**; no NYISO or
NREL PERFORM value is read or sampled by the code, and no parameter has been fitted to a
dataset. The datasets informed only the qualitative form (a diurnal load cycle with a
+/-20 % swing, small load noise, renewable forecast error). The experiments are
therefore the data-informed simulations, not a validation on the recorded data.


| Quantity | Model | Provenance |
|---|---|---|
| Demand D_t | 100N + 20N sin(2 pi t/48) + N(0, (0.8N)^2) MW | synthetic; diurnal shape informed by NYISO load |
| Supply noise w_it | N(0, 6^2) MW per agent and step | synthetic; stands for renewable forecast error |
| Initial supply / battery | U[80,120] MW / U[40,80] | synthetic |
| Supply cap, action range | 150 MW, +/-20 MW per step | synthetic design constants |
| Topology, limits | ring, b = 1, S_base = 10 MW, f_max = 2.5 pu | synthetic |

## Note on the latency

The latency reported for the SafeAgentGrid is the **mediation** time (proposals -> executed
action). The latency reported for the unmediated baseline techniques includes the sequential
inference of all of the  ten policies. `run_latency.py` measures every decision layer separately
in one process, so that the components are directly comparable......


## The main results (N = 10, 40 evaluation episodes)

| Metric | SafeAgentGrid | Best baseline |
|---|---|---|
| Avg. reward | 200,186.9 | 200,066.8 |
| Violations/step | 0.0125 +/- 0.0014 | 0.5754 +/- 0.0119 |
| Violations/step (20% FDIA) | 0.0194 +/- 0.0016 | 0.0941 +/- 0.0046 |
| Grid efficiency | 100.00% | 99.89% |
| Alignment | 0.999 | 0.982 |
| Fairness (Gini) | 0.046 | 0.090 |

The best baseline in every row is the joint safety shield, the strongest of the six
baselines implemented and trained in the same environment. Against the four unmediated
baselines, the Wilcoxon signed-rank tests over the 40 paired episodes give
p = 1.8 x 10^-12 (reward) and p <= 3.4 x 10^-8 (violations); eight tests, all favouring
SafeAgentGrid.


## Configuration note

All of the results in the paper utilize the consensus weight eta = 0.05. Section 4.13 of the paper
shows that eta = 0.50 is the recommended operating point, where the security module
decreases   the violations by a factor of 20 under FDIA. Set `SafeAgentGridMediator.ETA` to
change it.....




## The repository contents

| File | Purpose |
|---|---|
| `safeagentgrid.py` | full implementation: environment, agents, mediator, security module, training, evaluation |
| `run_*.py` | resumable scripts, one per table or sweep of the paper |
| `analyze_results.py` | statistics, figures and `results/summary.json` |
| `results/` | raw results of every run, including per-episode rewards and violation counts |
| `safeagentgrid_results.json` | metrics of the reference pipeline, including per-episode data |
| `REPRODUCIBILITY.md` | environment, what each script produces, and file hashes |
| `figures/` | the figures produced by the analysis scripts |



## The License

The implemented  code released under the MIT License; the manuscript is CC BY. Please contact
Qutaiba Alasad for more information.
