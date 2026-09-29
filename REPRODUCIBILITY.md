# THE Reproducibility record


Environment of the released results:

* software: torch 2.13.0, numpy 2.4.4, scipy 1.17.1 (`OMP_NUM_THREADS=1`)
* hardware: CPU-only, one core per process (Ubuntu 24); all latencies are wall-clock on
  that machine....

The pipeline is deterministic given the seeds. Training seed 42 is the reference model of
the paper, and every evaluation uses the same 40 episodes (evaluation seed 42). A
from-scratch run of `run_final40.py` reproduces the values of `safeagentgrid_results.json`
exactly (17 result sets, 0 mismatches; `analyze_results.py` performs this check
automatically). Latency is the one quantity that does not reproduce across machines.



## What each script will produce ....

| Script | Raw output | Content of the paper |
|---|---|---|
| `run_final40.py` | `results/final40.json` | Table 6 (main comparison, including the joint safety shield and the Lagrangian baseline), Table 9 (ablation), Table 12 (defended vs. undefended) |
| `run_scal40.py` | `results/scal40.json` (shipped copy: `safeagentgrid_results.json`, key `scalability_40ep`) | Table 10 (scalability) |
| `run_ci.py` | `results/ci_results.json` (shipped copy: `safeagentgrid_results.json`, key `sweeps_40ep`) | Tables 11 and 13 (attack severity, gate sweep) |
| `run_multiseed.py` | `results/multiseed/seed_*.json` | Table 7 and Figure 2 (ten independent training seeds) |
| `run_advantage.py --part A/B` | `results/advantage_A.json`, `results/advantage_B.json` | Table 8 (robustness across threat models) |
| `run_advantage_ms.py` | `results/advantage_ms/*.json` | Section 4.4, ten training seeds under Byzantine failures and the network-targeted FDIA |
| `run_chi_sweep.py` | `results/chi_sweep.json` | Table 17 and Figure 3 (tightening fraction chi; N = 16) |
| `run_weight_sensitivity.py` | `results/weight_sensitivity.json` | Table 18 (objective weights) |
| `run_training_weights.py` | `results/training_weights.json` | Table 19 (training-reward weights) |
| `run_conformal_coverage.py` | `results/conformal_coverage.json` | Table 15 (conformal coverage under distribution shift) |
| `run_sweeps40.py` | `results/sweeps40.json` | Tables 14 and 16 (stealth, delegation and repair sweeps) |
| `run_latency.py` | `results/latency.json` | like-for-like latency of every decision layer |
| `analyze_results.py` | `results/summary.json`, `figures/*.png` | every derived number and figure |


## THE Key numbers......

Reference model (N = 10, training seed 42, 40 evaluation episodes):


* SafeAgentGrid: 0.0125 +/- 0.0014 violations/step nominal, 0.0194 +/- 0.0016 under 20% FDIA, 0.0140 +/- 0.0015 under Byzantine failures; reward 200186.9
* joint safety shield: 0.5754 violations/step nominal, 0.0941 under 20% FDIA
* Lagrangian PG MARL: 0.9878 violations/step nominal, 0.6733 under 20% FDIA
* ten training seeds: SafeAgentGrid 0.0188 +/- 0.0174 (SD over seeds), joint safety shield 0.5829 +/- 0.0133
* robustness across threat models: SafeAgentGrid stays between 0.0140 and 0.0251 violations/step under the six attacks
* latency (one process, same machine): policy inference 4.98 ms, mediation 1.16 ms (16.7 ms on the full path only), joint safety shield 0.76 ms, box projection 0.063 ms



## A note on exact reproduction

The pipeline is deterministic on a given machine: repeated from-scratch runs
reproduce every reward and violation value bit-for-bit, and passing `seed=42`
explicitly is equivalent to the module-level seeding the scripts rely on.


The training is a 30-episode policy-gradient procedure, so a difference in CPU
instruction set or BLAS build can steer a re-run to a different policy on other
hardware. If your values differ from a single-seed table entry, compare them
against the ten-seed distribution reported in Section 4.3 of the paper
(`results/multiseed/`) rather than against the table entry itself.

The latency is wall-clock and varies by more than 20% between timings of the same
configuration, so it is not expected to reproduce.

## File hashes (SHA-256)

```
55268f202b30aff8d12b9d431f3a39128d06b8b82e55d1f6aac2aafc4c1d5849  results/chi_sweep.json
27287b6a5f898bd1e2bb59e8f594d8d53cc073f1ccf5fb85090d6e1359bde135  results/conformal_coverage.json
d388e96d5befd967b90bfd67a8c67c7edafea341fd1a2bdd0f3edba9ba207afc  results/final40.json
532ab24e0f344e94be8c892145948ac3e132e044aca10ec2cd707b628c273df2  results/latency.json
161213a4942fba7ca2eaa2252e6801ccfe63dfdc02436d9fec6fdb2273a4f814  results/sweeps40.json
982c325905830a6f513b111dc27a301b05c6f8a4639ce08aee62b7b61c5264e5  results/training_weights.json
bfc92610e2629fa8320368b3bac11a8b128f9ce068962aad677bf5e0a703b99e  results/weight_sensitivity.json
6e0709d6b6c53379f0f37a3aebfd4eea74d3c65f0b18393264e2990d22934cbc  results/multiseed/seed_42.json
fd3ab7a9e0ac330e3acf4c9773f077ccfc5cd7b6843e239472d9566da3a2710a  results/multiseed/seed_43.json
90dbbd03095507d9ea064c7e1f0bbd1a5c438861683d62807c81e1b1aaf30995  results/multiseed/seed_44.json
5b553d30917212cba51142dfd8069403f99a382f532bc730391abab274294378  results/multiseed/seed_45.json
571cdb119a822bc22ff9190aaee6a03248648804a8491fde5eda19e9ef2cad1b  results/multiseed/seed_46.json
0de5afa365b88189ea2bba3cbbe5e0a5e42dc69d43774c101bfd6395bc959534  results/multiseed/seed_47.json
f59dd1360eb5f265b85105bf91a633fc2e47762a65b0991102c0c73711373c5d  results/multiseed/seed_48.json
402b5498483bd020bd4a47120ae483a9ae5441a62a4d8759f5f79d7fefe39806  results/multiseed/seed_49.json
3e406cbcef3c3f66de95294fba0f68b6b0513043913b58ba7804a8feb99474c1  results/multiseed/seed_50.json
bb00055444ad7550fb1f83783580eacfde27ac9b534200eeda8664d792a6e50d  results/multiseed/seed_51.json
473dc27f8b067076cc1beb6240e7fe8353f62d0f63ff6e173fd2443d1fcbc013  results/advantage_A.json
90348ec942b91f733bd43a4ea4adc674923236343f71268ef399ec242b937b5b  results/advantage_B.json
103f4713e979d549336740de3124ebf1a66d305685f521ca58a6eb913a7e575d  results/advantage_ms/SAG_seed_43.json
adddde1d7f54fab0ef8e0cbdb85d844f7186870a82bdb2348aeedcdbd9b2d9a1  results/advantage_ms/SAG_seed_44.json
2642180ea1db2a2cbf131b77f1f23461d7638fbbc1e53b398efc97146692d4fd  results/advantage_ms/SAG_seed_45.json
48190d418cb08e987a2443d825758d80186aed716f884e42004f9892a51cf788  results/advantage_ms/SAG_seed_46.json
7645c702930e005a74ced6b2c71a92ee504234363f1a3bd04ea487f01dbf6d44  results/advantage_ms/SAG_seed_47.json
adfc31bf6ae17a104e62c25ca43ed50af2d24b7924c906971e15b2d4b60903b6  results/advantage_ms/SAG_seed_48.json
7b904d9a87cac8bb64d0944acac47bd347c6d563f5b8eabd0513f8b3600998ca  results/advantage_ms/SAG_seed_49.json
39848a76ce28f7d4c1b8f2a51177bddfda287e1a4a6db66c11bb76161c04d831  results/advantage_ms/SAG_seed_50.json
62865bb27f40bcab2d4f1d987766a5314d6c90bda1cd7bd2b88634bf259ec8f2  results/advantage_ms/SAG_seed_51.json
7fcad216f8cfb66d43ca178235aea338b70183fd76fe257917211092e86541b4  results/advantage_ms/shield_seed_42.json
d7c22b89dc6e00edf32d6cc6c55f426543f2ca3a6e5105e761b804ea0346e8b4  results/advantage_ms/shield_seed_43.json
f13254394b27c45876a89231911c2125c31be00e6168cc084af8c590ceff01b4  results/advantage_ms/shield_seed_44.json
4edd6ae69603c0b643b8a199d639475d5f21f20b5ab1c4911afbda1e2c5eafa3  results/advantage_ms/shield_seed_45.json
77484fdc0081dbe9b3e6760ac86642ec8b7fa83876c35ec6baf70d62d932eead  results/advantage_ms/shield_seed_46.json
fee4ce879ba4ceb168c315c55224b29d5e837473f913998e7293792ccf9bc957  results/advantage_ms/shield_seed_47.json
62c7d673b1b61d9b30bff787faff517fc43a1b762845ad0d0cd997fe9f024079  results/advantage_ms/shield_seed_48.json
fc36d7d30c965e99104a90fe57b8d9cf0b68db0005648a36db93720b7cd63fcc  results/advantage_ms/shield_seed_49.json
c4d392edcdb42a8a8e0fe5d0812e5bc9fbc7779ad224f55265b53d01bef06c73  results/advantage_ms/shield_seed_50.json
b22818fff08d1724ee181b66c9ee1a8ac562ecdc646ef585a4245d23dc44a224  results/advantage_ms/shield_seed_51.json
5f5e505d8c152708c7e9166e4f2bda42c9120641273f8e839484747a0120a149  results/summary.json
788f2ce247d326da7f07ce65548a3491e94830949a19e77675a0ed583129bdec  safeagentgrid_results.json
7067890265e5d02b3a927723280722458ccbf12e54811ac2a5cebb294e503bc5  safeagentgrid.py
```
