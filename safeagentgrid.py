# =====================================================================
# THE SafeAgentGrid - reference implementation.............

# The hierarchical constitutional multi-agent framework for trustworthy resource
# optimisation in smart energy infrastructures:
#   Layer 1  graph-augmented constitutional agents with a projection filter
#            that enforces the hard box-type principles by construction
#   Layer 2  hybrid neural-surrogate mediator: a conformally gated fast path
#            and an SLSQP full path over a tightened DC reachable set, with a
#            security module (MAD anomaly detection, reference-anchored repair
#            and risk-aware security weighting)
#   Layer 3  8-bit quantised models for edge deployment


# The environment is a DC power-flow simulator on a ring-topology susceptance
# matrix.  Demand and supply are synthetic parametric processes; no external
# dataset is read by this code (see the data-provenance note at the demand
# model).....

# Usage:  python safeagentgrid.py --medium
# =========================#####======================================



import json
import argparse
import time
import warnings
from collections import deque
from typing import Dict, List, Optional, Tuple


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize
from scipy.stats import wilcoxon

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
DEVICE = torch.device("cpu")





# =======------------============== 1. THE  ENVIRONMENT (DC POWER FLOW) =====================

class SmartGridEnv:
    """Ring-topology grid with linearized DC power flow.

    Documented constants:
        S_base   = 10 MW   per-unit power base
        f_max    = 2.5 pu  line-flow limit
        b_line   = 1.0     per-line susceptance (ring Laplacian, rows sum to 0)
        slack    = bus 0 (theta_0 = 0)
        voltage proxy: V_i = 1 + 0.005 * theta_i, limit |V_i - 1| <= 0.05
    Demand base is 100 MW/agent so that installed supply (<=150 MW/agent)
        and demand are commensurate and the efficiency metric can vary.
    """

    S_BASE = 10.0
    F_MAX = 2.5
    V_SLOPE = 0.005
    V_TOL = 0.05
    SUPPLY_CAP = 150.0
    SHORTFALL_TOL = 5.0

    def __init__(self, N: int = 10, T: int = 200, kappa: float = 0.05):
        self.N = N
        self.T = T
        self.kappa = kappa
        b_line = 1.0
        B = np.zeros((N, N))
        for i in range(N):
            j = (i + 1) % N
            B[i, i] += b_line
            B[j, j] += b_line
            B[i, j] -= b_line
            B[j, i] -= b_line
        self.B = B
        self.Binv = np.linalg.inv(B[1:, 1:])          # cached reduced inverse
        self.lines = [(i, (i + 1) % N) for i in range(N)]
        self._li = np.array([i for (i, j) in self.lines])
        self._lj = np.array([j for (i, j) in self.lines])
        self._lb = np.array([self.B[i, j] for (i, j) in self.lines])
        # Precomputed linear maps: the DC flow model is affine in the supply
        # vector, so theta = Th @ supply and f = A @ supply.  Caching these
        # turns each objective evaluation into two mat-vecs instead of a solve.
        C = np.eye(N) - np.ones((N, N)) / N          # mean-centering
        E = np.zeros((N, N)); E[1:, 1:] = self.Binv  # slack bus 0
        G = np.zeros((N, N))
        for k, (i, j) in enumerate(self.lines):
            G[k, i] -= self.B[i, j]; G[k, j] += self.B[i, j]
        self._Th = E @ C / self.S_BASE
        self._A = G @ self._Th
        self.f_max = self.F_MAX



    # ---==- demand: 100N + 20N sin(2 pi t / 48) + N(0, 0.8N)
    # DATA PROVENANCE.  Demand and supply are SYNTHETIC parametric
    # processes; no NYISO / NREL PERFORM value is read or sampled by this code.
    #   demand : 100 MW/agent base, +/-20 % diurnal swing with a 48-step period
    #            (one day at 30-min resolution), Gaussian noise with standard
    #            deviation 0.8N MW (np.random.normal takes the STD, not the
    #            variance).  The daily-cycle SHAPE was chosen to mimic the
    #            public NYISO load / PERFORM traces; it is not fitted to them.
    #   supply : per-agent Gaussian realisation noise, standard deviation
    #            6 MW, added AFTER the action (see step()).
    #   initial supply U[80,120] MW, initial battery U[40,80] %.
    # The experiments are therefore data-informed simulations, not a
    # validation on recorded data.
    def _demand_at(self, t: int) -> float:
        return (100.0 * self.N
                + 20.0 * self.N * np.sin(2 * np.pi * t / 48)
                + np.random.normal(0, 0.8 * self.N))

    def reset(self):
        self.t = 0
        self.supply = np.random.uniform(80, 120, self.N).astype(np.float64)
        self.battery = np.random.uniform(40, 80, self.N).astype(np.float64)
        self.demand = self._demand_at(0)
        return self._get_state()

    def _get_state(self):
        return np.concatenate(
            [self.supply / self.SUPPLY_CAP,
             self.battery / 100.0,
             [self.demand / (120.0 * self.N)]]
        ).astype(np.float32)



    def preview_flows(self, supply: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self._Th @ supply, self._A @ supply

    def local_rewards(self, supply_next: np.ndarray, actions: np.ndarray,
                      allocated: float) -> np.ndarray:
        """Per-agent reward r_{i,t}: allocated energy share minus
        operational cost.  Used for the Gini coefficient of Eq. (18)."""
        total = float(np.sum(supply_next))
        share = allocated / total if total > 1e-9 else 0.0
        return supply_next * share - 0.1 * np.abs(actions)

    def step(self, actions: np.ndarray):
        actions = np.clip(np.asarray(actions, dtype=np.float64), -20, 20)
        noise = np.random.normal(0, 6, self.N)
        self.supply = np.clip(self.supply + actions + noise, 0, self.SUPPLY_CAP)

        total_available = float(np.sum(self.supply))
        allocated = min(total_available, self.demand)
        shortfall = max(0.0, self.demand - total_available)
        excess = total_available - allocated
        self.battery = np.clip(self.battery + excess / self.N, 0, 100)

        theta, f = self.preview_flows(self.supply)
        voltage_viol = bool(np.any(np.abs(self.V_SLOPE * theta) > self.V_TOL))
        line_viol = bool(np.any(np.abs(f) > self.f_max))
        line_overload = float(np.sum(np.maximum(0.0, np.abs(f) - self.f_max)))

        demand_t = self.demand
        r_local = self.local_rewards(self.supply, actions, allocated)
        self.demand = self._demand_at(self.t + 1)
        self.t += 1
        done = self.t >= self.T

        reward = (allocated - 0.5 * shortfall
                  - 0.1 * float(np.sum(np.abs(actions)))
                  - self.kappa * line_overload)
        violation = int((shortfall > self.SHORTFALL_TOL) or voltage_viol or line_viol)

        info = {
            "violation": violation,
            "viol_shortfall": int(shortfall > self.SHORTFALL_TOL),
            "viol_voltage": int(voltage_viol),
            "viol_line": int(line_viol),
            "efficiency": allocated / max(demand_t, 1e-6),
            "gini": self._gini(r_local),          # Gini over rewards, Eq. (18)
            "local_rewards": r_local,
            "max_flow": float(np.max(np.abs(f))),
            # continuous constraint-violation magnitudes, used as
            # the cost signal of the Lagrangian constrained-MARL baseline
            "line_overload": line_overload,
            "shortfall": shortfall,
        }
        return self._get_state(), reward, done, info

    @staticmethod
    def _gini(x):
        """Gini coefficient, computed with the sorted-rank identity (O(n log n))
        rather than the O(n^2) pairwise form; the two agree exactly."""
        x = np.sort(np.asarray(x, dtype=np.float64))
        x = x - min(0.0, float(x[0]))             # Gini needs non-negative values
        n = x.size
        tot = x.sum()
        if tot <= 1e-8:
            return 0.0
        idx = np.arange(1, n + 1)
        return float((2.0 * np.dot(idx, x)) / (n * tot) - (n + 1.0) / n)


# ===================== 2. AGENTS + GAT ===============--------------=================



class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 32):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.att = nn.Linear(2 * out_dim, 1)

    def forward(self, x):
        N = x.shape[0]
        x = self.fc(x)
        x_i = x.unsqueeze(1).repeat(1, N, 1)
        x_j = x.unsqueeze(0).repeat(N, 1, 1)
        att = F.softmax(self.att(torch.cat([x_i, x_j], dim=-1)).squeeze(-1), dim=-1)
        return torch.matmul(att, x) + x





class AgentPolicy(nn.Module):
    """Stochastic Gaussian policy so that Eq. (22) is the true REINFORCE
    estimator.  `deterministic=True` returns the mean (used at evaluation)."""

    def __init__(self, state_dim: int, use_gat: bool = True, N: int = 10):
        super().__init__()
        self.use_gat = use_gat
        self.N = N
        if use_gat:
            self.gat = GATLayer(state_dim, 32)
            mlp_in = 32
        else:
            mlp_in = state_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.log_std = nn.Parameter(torch.tensor(0.0))

    def _mean(self, state):
        x = torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        if self.use_gat:
            x = x.repeat(self.N, 1)
            x = self.gat(x)
            x = x.mean(dim=0, keepdim=True)
        return 20.0 * torch.tanh(self.mlp(x).squeeze() / 20.0)

    def forward(self, state):
        return self._mean(state)

    def act(self, state, deterministic: bool = False):
        mu = self._mean(state)
        if deterministic:
            return mu.detach(), None
        std = torch.exp(self.log_std).clamp(0.05, 5.0)
        dist = torch.distributions.Normal(mu, std)
        a = dist.sample()
        return a.detach(), dist.log_prob(a)

    def log_prob(self, state, action):
        mu = self._mean(state)
        std = torch.exp(self.log_std).clamp(0.05, 5.0)
        return torch.distributions.Normal(mu, std).log_prob(
            torch.as_tensor(action, dtype=torch.float32))





class ValueBaseline(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, state):
        x = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        return self.net(x).squeeze(-1)


# =============----------=========== 3. THE SECURITY   MODULE =====================

class SecurityModule:
    """Per-agent temporal robust z-score (MAD), rolling window 60, tau = 3,
    scale floored at 0.5 action units, 10-sample unconditional warm-up."""

    WARMUP = 10
    HISTLEN = 60

    def __init__(self, enabled: bool = True, tau: float = 3.0):
        self.enabled = enabled
        self.tau = tau
        self._buf = None
        self._cnt = None
        self._pos = None

    def reset(self):
        self._buf = None
        self._cnt = None
        self._pos = None

    def _cross_sectional(self, proposals: np.ndarray) -> np.ndarray:
        med = np.median(proposals)
        mad = np.median(np.abs(proposals - med))
        scale = max(1.4826 * mad, 0.5)
        return np.abs(proposals - med) / scale > self.tau

    def _rowmedian(self, M):
        """Row-wise median over the valid prefix of each row.  NaN padding is
        pushed to the end by sorting, then the median index is taken from the
        per-row valid count.  ~20x faster than np.nanmedian at this size."""
        X = np.where(np.isnan(M), np.inf, M)
        X.sort(axis=1)
        c = np.maximum(self._cnt, 1)
        lo = X[np.arange(M.shape[0]), (c - 1) // 2]
        hi = X[np.arange(M.shape[0]), c // 2]
        return 0.5 * (lo + hi)

    def flag_anomalies(self, proposals: np.ndarray) -> np.ndarray:
        """Vectorised over agents: the per-agent medians and MADs are computed
        with a single nanmedian over the (N, HISTLEN) circular buffer rather
        than an N-iteration Python loop.  Behaviour is unchanged."""
        proposals = np.asarray(proposals, dtype=np.float64)
        n = len(proposals)
        if not self.enabled or n < 3:
            return np.zeros(n, dtype=bool)
        if self._buf is None or self._buf.shape[0] != n:
            self._buf = np.full((n, self.HISTLEN), np.nan)
            self._cnt = np.zeros(n, dtype=int)
            self._pos = np.zeros(n, dtype=int)

        warm = self._cnt >= self.WARMUP
        flags = np.zeros(n, dtype=bool)
        if warm.any():
            med_i = self._rowmedian(self._buf)
            mad_i = self._rowmedian(np.abs(self._buf - med_i[:, None]))
            scale_i = np.maximum(1.4826 * mad_i, 0.5)
            flags[warm] = (np.abs(proposals - med_i)[warm]
                           / scale_i[warm]) > self.tau
        if (~warm).any():
            med = np.median(proposals)
            mad = np.median(np.abs(proposals - med))
            scale = max(1.4826 * mad, 0.5)
            flags[~warm] = (np.abs(proposals - med) / scale)[~warm] > self.tau

        upd = (self._cnt < self.WARMUP) | (~flags)
        idx = np.nonzero(upd)[0]
        if idx.size:
            self._buf[idx, self._pos[idx]] = proposals[idx]
            self._pos[idx] = (self._pos[idx] + 1) % self.HISTLEN
            self._cnt[idx] = np.minimum(self._cnt[idx] + 1, self.HISTLEN)
        return flags

    def severity(self, proposals: np.ndarray, flags: np.ndarray) -> float:
        if not flags.any():
            return 0.0
        med = np.median(proposals[~flags]) if (~flags).any() else np.median(proposals)
        dev = np.mean(np.abs(proposals[flags] - med))
        return float(min(1.0, dev / 40.0))



    def robust_aggregate(self, proposals: np.ndarray, flags: np.ndarray) -> np.ndarray:
        if not flags.any():
            return proposals
        clean = proposals[~flags] if (~flags).any() else proposals
        out = proposals.copy()
        out[flags] = np.median(clean)
        return out





# ============== 4. THE   CONSTITUTION (Eq.    21) ========  -----=============


class Constitution:
    """Hard box + soft principles, with an explicit principle library and
    online selection ranked by the alignment improvement each candidate yields."""

    LIBRARY = {
        "tighter_action_box": dict(a_min=-15.0, a_max=15.0),
        "higher_reserve_margin": dict(reserve=0.08),
        "stronger_fairness": dict(nu=0.6),
    }
    # floor of the dynamic security weight: rho_t = RHO0 +
    # (1 - RHO0) * severity.  RHO0 = 0.2 reproduces rho_t = 0.2 + 0.8 * severity.
    RHO0 = 0.2

    def __init__(self):
        self.a_min, self.a_max = -20.0, 20.0
        self.nu = 0.3              # fairness (Gini) weight in the mediator objective
        self.beta = 0.3            # soft-principle weight in the SHAPED REWARD
        self.rho = self.RHO0       # dynamic security weight  
        self.reserve = 0.05        # required operating reserve margin
        self.theta_risk = 0.7
        self.active_extra: List[str] = []
        self._tightened_until = -1

    def _defaults(self):
        self.a_min, self.a_max = -20.0, 20.0
        self.nu = 0.3
        self.reserve = 0.05
        self.rho = self.RHO0
        self.active_extra = []

    def adapt(self, risk: float, t: int, ranker=None):
        """Eq. (21): C_{t+1} = C_t union {c_new : rho_t > theta_risk}, where the
        new principle is the library entry with the highest ranked alignment."""
        if risk > self.theta_risk:
            best = "tighter_action_box"
            if ranker is not None:
                scores = {}
                for name, params in self.LIBRARY.items():
                    scores[name] = ranker(name, params)
                best = max(scores, key=scores.get)
            self._defaults()
            for k, v in self.LIBRARY[best].items():
                setattr(self, k, v)
            self.rho = self.RHO0 + (1.0 - self.RHO0) * risk
            self.active_extra = [best]
            self._tightened_until = t + 10
        elif t > self._tightened_until:
            self._defaults()

    def bounds(self) -> Tuple[float, float]:
        """Bounds of the ACTIVE action box B_t.  Every hard box-type
        principle contributes an interval; B_t is their intersection.  The
        nominal principle is the actuator/ramp limit [-20, 20] MW per step; the
        library principle 'tighter_action_box' narrows it to [-15, 15] while it
        is active.  (With nested intervals the intersection is the innermost.)"""
        return self.a_min, self.a_max

    def filter(self, a: np.ndarray) -> np.ndarray:
        """Constitutional filter Phi_{C_t}: Euclidean projection onto
        the active box B_t = [a_min, a_max]^N,
            Phi(a) = argmin_{x in B_t} ||x - a||_2 .
        The problem is separable over agents; the KKT conditions of each scalar
        problem give x_i = min(max(a_i, a_min), a_max), i.e. coordinate-wise
        clipping.  Phi is idempotent, non-expansive (1-Lipschitz), costs O(N),
        has no trainable parameters, and its output lies in B_t for ANY input
        (Proposition 1).  Non-finite entries (NaN/+-Inf, e.g. a malformed or
        malicious message) are first mapped into the box, so the guarantee
        also covers them; for finite inputs this step is the identity and the
        results are unchanged."""
        a = np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0,
                          posinf=self.a_max, neginf=self.a_min)
        return np.clip(a, self.a_min, self.a_max)


# ===================== 5. THE MEDIATOR ========---------------=============


class FastSurrogate(nn.Module):
    def __init__(self, state_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, act_dim),
        )

    def forward(self, state):
        x = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        y = self.net(x)
        return y.squeeze(0) if squeeze else y


class SafeAgentGridMediator:
    """Hybrid neural-surrogate mediator implementing Eq. (14)-(20)."""

    ETA = 0.05      # consensus weight (eta)
    LAM = 5.0       # constitutional-distance weight (lambda)
    MU = 50.0       # reachability weight (mu): line and voltage limits are hard
                    # operating constraints and must dominate the consensus and
                   # fairness preferences
    TIGHTEN = 0.5   # chi: fraction of the nominal limit enforced by the
                       # mediator, reserving the remainder for the realisation
                    # noise between decision and execution

    def __init__(self, N: int, state_dim: int, security_enabled: bool = True,
                 fast_path_enabled: bool = True, repair: str = "reference"):
        self.N = N
        self.repair = repair
        # the surrogate imitates the mediator, whose input is the
        # STATE *and* the aggregated proposals; conditioning on the state alone
        # leaves an irreducible target variance that no amount of training can
        # remove.
        self.fast_surrogate = FastSurrogate(state_dim + N, N).to(DEVICE)
        self.security = SecurityModule(security_enabled)
        self.constitution = Constitution()
        self.opt = optim.Adam(self.fast_surrogate.parameters(), lr=1e-3)
        self.surrogate_trained = False
        self.fast_path_enabled = fast_path_enabled
        self.nu_scale = 1.0
        self.rho_scale = 1.0       # multiplier on rho_t (sweep only)
        # optional SLSQP telemetry: list of (success, nit) per solve
        self.solver_log: Optional[List[Tuple[bool, int]]] = None
        # optional shadow log: on EVERY step, the alignment the
        # filtered fast-path action WOULD have had, whether or not the fast
        # path was taken.  It adds no randomness and does not alter decisions.
        self.shadow_log: Optional[List[Tuple[float, int, int]]] = None
        self.fast_gate = 0.99
        self.last_alignment = 1.0
        self.conformal_q: Optional[float] = None
        self.conformal_delta = 0.10
        self.fast_latencies: List[float] = []
        self.full_latencies: List[float] = []
        self._step_counter = 0
        self.fast_path_uses = 0
        self.total_mediations = 0
        self.anomaly_steps = 0



    def reset_telemetry(self):
        """latency accumulators must not span training + every sweep."""
        self.fast_latencies = []
        self.full_latencies = []
        self.fast_path_uses = 0
        self.total_mediations = 0
        self.anomaly_steps = 0

    # ---- Eq. (20): fraction of satisfied constitutional constraints
    def alignment_score(self, a: np.ndarray, env: SmartGridEnv) -> float:
        c = self.constitution
        checks: List[bool] = []
        checks += [abs(v) <= c.a_max + 1e-6 for v in a]
        sup_next = np.clip(env.supply + a, 0, env.SUPPLY_CAP)
        theta, f = env.preview_flows(sup_next)
        checks += [abs(fl) <= env.f_max for fl in f]
        checks += [abs(env.V_SLOPE * th) <= env.V_TOL for th in theta]
        shortfall = max(0.0, env.demand - float(np.sum(sup_next)))
        checks.append(shortfall <= env.SHORTFALL_TOL)
        checks.append(float(np.sum(sup_next)) >= (1.0 + c.reserve) * env.demand)
        alloc = min(float(np.sum(sup_next)), env.demand)
        checks.append(SmartGridEnv._gini(env.local_rewards(sup_next, a, alloc)) <= 0.35)
        return float(np.mean(checks))

    # ---- Eq. (15)-(19) terms
    def _objective_factory(self, env, p_agg, flags, ref_vec, nu, rho):
        c = self.constitution
        sup0 = env.supply
        demand = env.demand
        cap = env.SUPPLY_CAP
        N = self.N
        fmax = env.f_max

        def objective(a):
            sup_next = np.clip(sup0 + a, 0, cap)
            total = float(np.sum(sup_next))
            shortfall = max(0.0, demand - total)
            allocated = min(total, demand)

            # Eq. (15) cost: unmet demand + operating cost, per agent
            cost = shortfall / N + 0.1 * float(np.mean(np.abs(a)))
            # consensus with the robustly aggregated proposal
            consensus = self.ETA * float(np.mean((a - p_agg) ** 2))
            # Eq. (16) constitutional distance
            dist = (max(0.0, (1.0 + c.reserve) * demand - total) / N
                    + float(np.sum(np.maximum(0.0, np.abs(a) - c.a_max))))
            # Eq. (17) DC power-flow reachability, evaluated on a TIGHTENED
            # feasible set.  The mediator decides before the per-step supply
            # deviation w ~ N(0, sigma) is realised, so enforcing the nominal
            # limit leaves solutions sitting exactly on the boundary that the
            # realisation then crosses.  We therefore enforce |f| <= chi * f_max
            # with chi = TIGHTEN < 1 and use a squared hinge, which is smooth
            # for the SLSQP finite-difference Jacobian and pushes the solution
            # into the interior of the reachable set.
            theta, f = env.preview_flows(sup_next)
            reach = (float(np.sum(np.maximum(0.0,
                                             np.abs(f) - self.TIGHTEN * fmax) ** 2))
                     + float(np.sum(np.maximum(
                         0.0, np.abs(env.V_SLOPE * theta)
                         - self.TIGHTEN * env.V_TOL) ** 2)))
          # Eq. (18) Gini over per-agent rewards
            gini = SmartGridEnv._gini(env.local_rewards(sup_next, a, allocated))
            # Eq. (19) squared-L2 security penalty on flagged proposals
            sec = (rho * float(np.sum((a[flags] - ref_vec[flags]) ** 2))
                   if flags.any() else 0.0)
            return (cost + consensus + self.LAM * dist + self.MU * reach
                    + nu * gini + sec)

        return objective



    def mediate(self, proposals, state, env, is_high_risk: bool = False):
        start = time.perf_counter()
        self.total_mediations += 1
        p = np.asarray(proposals, dtype=np.float64)

        flags = self.security.flag_anomalies(p)
        if flags.any():
            self.anomaly_steps += 1
        risk = self.security.severity(p, flags)
        self._step_counter += 1

        # Repair reference.  Median-of-proposals repair implicitly assumes
        # the honest agents are EXCHANGEABLE in action space.  They are not: each
        # bus sits at a different supply level, so the correct action for a
         # flagged agent is not the population median but the action that returns
         # ITS OWN bus to the common operating point.  Substituting the median
       # therefore injects a physically wrong action at exactly the buses under
      # attack.  `repair="reference"` substitutes the model-based per-agent
        # reference instead, and anchors the Eq. (19) penalty on it.
        target = (1.0 + self.constitution.reserve) * env.demand / self.N
        ref_vec = np.clip(target - env.supply, self.constitution.a_min,
                          self.constitution.a_max)
        if self.repair == "reference":
            p_agg = p.copy()
            if flags.any():
                p_agg[flags] = ref_vec[flags]
        else:
            p_agg = self.security.robust_aggregate(p, flags)
            ref_vec = np.full(self.N, float(np.median(p_agg)))

      # rank library candidates by the alignment of the filtered aggregate
        def _rank(name, params):
            saved = {k: getattr(self.constitution, k) for k in params}
            for k, v in params.items():
                setattr(self.constitution, k, v)
            score = self.alignment_score(self.constitution.filter(p_agg), env)
            for k, v in saved.items():
                setattr(self.constitution, k, v)
            return score

        self.constitution.adapt(risk, self._step_counter,
                                ranker=_rank if risk > self.constitution.theta_risk
                                else None)

        # shadow fast-path alignment (diagnostic only; deterministic,
        # consumes no random numbers, never changes the executed action)
        shadow_al = None
        if self.shadow_log is not None and self.surrogate_trained:
            feat_s = np.concatenate([np.asarray(state, dtype=np.float32),
                                     p_agg.astype(np.float32)])
            with torch.no_grad():
                a_s = self.fast_surrogate(feat_s).cpu().numpy().astype(np.float64)
            shadow_al = self.alignment_score(self.constitution.filter(a_s), env)

        # ---- the fast path .............
        if (self.fast_path_enabled and self.surrogate_trained
                and not is_high_risk and not flags.any()
                and self.last_alignment >= self.fast_gate
                and self.conformal_q is not None
                and self.conformal_q <= 0.05):
            feat = np.concatenate([np.asarray(state, dtype=np.float32),
                                   p_agg.astype(np.float32)])
            with torch.no_grad():
                a = self.fast_surrogate(feat).cpu().numpy().astype(np.float64)
            a = self.constitution.filter(a)
            latency = (time.perf_counter() - start) * 1000
            self.fast_path_uses += 1
            self.fast_latencies.append(latency)
            self.last_alignment = self.alignment_score(a, env)
            if shadow_al is not None:
                self.shadow_log.append((shadow_al, 1, int(flags.any())))
            return a, latency, self.last_alignment

        # ---- full path (Eq. 14) ,,, .....
        nu = self.constitution.nu * self.nu_scale
        objective = self._objective_factory(env, p_agg, flags, ref_vec,
                                            nu, self.constitution.rho
                                            * self.rho_scale)
        res = minimize(objective, p_agg, method="SLSQP",
                       bounds=[(self.constitution.a_min,
                                self.constitution.a_max)] * len(p),
                       options={"maxiter": 40, "ftol": 1e-4})
        if self.solver_log is not None:
            self.solver_log.append((bool(res.success), int(res.nit)))
        a = self.constitution.filter(res.x)
        self.last_alignment = self.alignment_score(a, env)

        if self.last_alignment < 0.95:
            a = 0.5 * a + 0.5 * np.clip(p_agg, self.constitution.a_min,
                                        self.constitution.a_max)
            a = self.constitution.filter(a)
            self.last_alignment = self.alignment_score(a, env)

        latency = (time.perf_counter() - start) * 1000
        self.full_latencies.append(latency)
        if shadow_al is not None:
            self.shadow_log.append((shadow_al, 0, int(flags.any())))
        return a, latency, self.last_alignment

    # ---- proper   mini-batch imitation training, conditioned on proposals
    @staticmethod
    def _sample_proposals(agents, s, N, explore=0.0):
        if agents is None:
            return np.random.uniform(-18, 18, N)
        p = np.array([float(a.act(s, deterministic=True)[0]) for a in agents])
        if explore > 0:
            p = p + np.random.normal(0, explore, N)
        return np.clip(p, -20, 20)

    def train_surrogate(self, env, agents=None, num_collections: int = 40,
                        horizon: int = 40, epochs: int = 200,
                        batch_size: int = 64, verbose=True):
        if verbose:
            print("Training FastSurrogate via imitation learning...")
        feats, targets = [], []
        for _ in range(num_collections):
            s = env.reset()
            for _ in range(horizon):
                props = self._sample_proposals(agents, s, env.N, explore=8.0)
                feats.append(np.concatenate([s.copy(), props]))
                a, _, _ = self.mediate(props.tolist(), s, env, is_high_risk=True)
                targets.append(a.copy())
                env.step(a)
                s = env._get_state()
        S = torch.tensor(np.array(feats), dtype=torch.float32)
        Tg = torch.tensor(np.array(targets), dtype=torch.float32)

        def _fit(S, Tg, epochs):
            ds = torch.utils.data.TensorDataset(S, Tg)
            dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            last = float("nan")
            for ep in range(epochs):
                tot, nb = 0.0, 0
                for xb, yb in dl:
                    loss = F.mse_loss(self.fast_surrogate(xb), yb)
                    self.opt.zero_grad(); loss.backward(); self.opt.step()
                    tot += loss.item(); nb += 1
                last = tot / max(nb, 1)
                if verbose and ep % max(1, epochs // 5) == 0:
                    print(f"  Surrogate epoch {ep:3d} | MSE: {last:.4f}")
            return last

        mse = _fit(S, Tg, epochs)
        self.surrogate_trained = True
        if verbose:
            print(f"  final imitation MSE: {mse:.4f}")

        for rnd in range(4):
            self.calibrate_conformal(env, agents)
            if self.conformal_q <= 0.05:
                break
            if verbose:
                print(f"  [train-to-qualify] round {rnd+1}: q={self.conformal_q:.4f}")
            ex_f, ex_t = [], []
            for _ in range(max(10, num_collections // 2)):
                s = env.reset()
                for _ in range(horizon):
                    props = self._sample_proposals(agents, s, env.N, explore=4.0)
                    feat = np.concatenate([s.copy(), props])
                    with torch.no_grad():
                        a_sur = self.constitution.filter(
                            self.fast_surrogate(feat.astype(np.float32)).cpu().numpy())
                    a_lab, _, _ = self.mediate(props.tolist(), s, env,
                                               is_high_risk=True)
                    ex_f.append(feat); ex_t.append(a_lab.copy())
                    env.step(a_sur)
                    s = env._get_state()
            S = torch.cat([S, torch.tensor(np.array(ex_f), dtype=torch.float32)])
            Tg = torch.cat([Tg, torch.tensor(np.array(ex_t), dtype=torch.float32)])
            _fit(S, Tg, max(20, epochs // 2))
        if verbose:
            adm = "ADMISSIBLE" if self.conformal_q <= 0.05 else "DISABLED"
            print(f"FastSurrogate trained. Conformal q(delta="
                  f"{self.conformal_delta:.2f}) = {self.conformal_q:.4f} "
                  f"-> fast path {adm}.\n")

    def calibrate_conformal(self, env, agents=None, m: int = 150):
        """Split-conformal calibration on a fresh trajectory driven by the
        DEPLOYED proposal distribution."""
        scores = []
        s = env.reset()
        with torch.no_grad():
            for _ in range(m):
                props = self._sample_proposals(agents, s, env.N)
                feat = np.concatenate([s, props]).astype(np.float32)
                a = self.constitution.filter(self.fast_surrogate(feat).cpu().numpy())
                scores.append(1.0 - self.alignment_score(a, env))
                env.step(a)
                s = env._get_state()
        scores = np.sort(np.asarray(scores))
        k = min(m, int(np.ceil((1.0 - self.conformal_delta) * (m + 1))))
        self.conformal_q = float(scores[k - 1])
        return self.conformal_q






# ========-----= 6. THE ATTACKS  ,  ===----==================

def apply_fdia(proposals, env, frac: float = 0.20, lo: float = 20.0,
               hi: float = 35.0):
    """FDIA: corrupt floor(frac*N) agents with adversarially signed deltas.
    lo/hi expose the perturbation magnitude for the stealth sweep."""
    p = list(proposals)
    k = max(1, int(np.floor(frac * env.N)))
    idx = np.random.choice(env.N, size=k, replace=False)
    gap = env.demand - float(np.sum(env.supply))
    sign = -np.sign(gap) if gap != 0 else np.random.choice([-1.0, 1.0])
    for i in idx:
        p[i] += sign * np.random.uniform(lo, hi)
    return p


def apply_fdia_flow(proposals, env, frac: float = 0.20, lo: float = 20.0,
                    hi: float = 35.0):
    """Network-targeted FDIA (load-redistribution style).  The same
    floor(frac*N) randomly chosen agents are corrupted as in apply_fdia, but the
    sign of each perturbation is chosen to INCREASE the flow on the currently
    most loaded line (sign of the flow times the sign of the agent's flow
    sensitivity), i.e. the attacker targets line overloads instead of the
    supply-demand balance.  Magnitudes are |delta| ~ U[lo, hi] as in apply_fdia."""
    p = list(proposals)
    k = max(1, int(np.floor(frac * env.N)))
    idx = np.random.choice(env.N, size=k, replace=False)
    f = env._A @ env.supply
    line = int(np.argmax(np.abs(f)))
    for i in idx:
        sign = float(np.sign(f[line]) * np.sign(env._A[line, i]))
        if sign == 0.0:
            sign = float(np.random.choice([-1.0, 1.0]))
        p[i] += sign * np.random.uniform(lo, hi)
    return p


def apply_byzantine(proposals, env):
    p = list(proposals)
    k = env.N // 3
    idx = np.random.choice(env.N, size=k, replace=False)
    for i in idx:
        p[i] = float(np.random.uniform(-60, 60))
    return p


# =============== 7. THE TRAINING HERE ------=====================


def train_safeagentgrid(episodes: int, N: int, surrogate_collections: int,
                        use_gat: bool = True, verbose: bool = True,
                        surr_epochs: int = 200, seed: Optional[int] = None,
                        beta: Optional[float] = None, lam_sec: float = 0.01):
    """Mediator-in-the-loop REINFORCE with a learned baseline, Eq. (22).
    Because the mediator objective now anchors on the agents' proposals
    (consensus term), the executed action depends on the policies even when
    every step takes the full path, so the policy gradient carries signal.
    The surrogate is trained afterwards on the DEPLOYED proposal distribution
    so that its conformal certificate is calibrated for deployment.

    seed : if given, numpy and torch are re-seeded before training,
               so that independent training runs can be replicated.  seed=42
               reproduces the reference model of the paper bit-for-bit (the
               module-level seeding uses 42 and nothing consumes random numbers
               before the first training call).
    beta, lam_sec : weights of the training reward of Eq. (27) of the paper,
               r_t - beta * Gini_t - lam_sec * ||a'_t - a*_t||^2 / N.
               Defaults (beta = Constitution.beta = 0.3, lam_sec = 0.01)
               reproduce the paper."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    state_dim = 2 * N + 1
    env = SmartGridEnv(N=N)
    agents = [AgentPolicy(state_dim, use_gat=use_gat, N=N) for _ in range(N)]
    optimizers = [optim.Adam(a.parameters(), lr=3e-4) for a in agents]
    critic = ValueBaseline(state_dim)
    critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
    mediator = SafeAgentGridMediator(N, state_dim, security_enabled=True)
    gamma = 0.99

    if verbose:
        print(f"Training SafeAgentGrid: {episodes} mediator-in-the-loop "
              f"episodes, N={N}")
    curve = []
    for ep in range(episodes):
        s = env.reset()
        S, R, LOGP, SHAPE = [], [], [], []
        ep_reward = 0.0
        for _ in range(env.T):
            logps, props = [], []
            for i in range(N):
                a_i, lp = agents[i].act(s)
                props.append(float(a_i)); logps.append(lp)
            props = list(mediator.constitution.filter(np.asarray(props)))
            a, _, _ = mediator.mediate(props, s, env)
            s2, r, done, info = env.step(a)
            # Eq. (13)+(23): shaped reward and security term
            # = Eq. (27) of the paper: fixed-multiplier
            # Lagrangian relaxation of the soft (fairness) principle and of the
            # "agreement with the constitutional mediator" constraint.
            sec_pen = float(np.mean((np.asarray(props) - a) ** 2))
            b_w = mediator.constitution.beta if beta is None else beta
            shaped = -b_w * info["gini"] - lam_sec * sec_pen
            S.append(s.copy()); R.append(r); LOGP.append(torch.stack(logps))
            SHAPE.append(shaped)
            ep_reward += r
            s = s2
            if done:
                break

        Rtot = [r + sh for r, sh in zip(R, SHAPE)]
        G, returns = 0.0, []
        for r in reversed(Rtot):
            G = r + gamma * G
            returns.insert(0, G)
        ret_t = torch.tensor(returns, dtype=torch.float32)
        S_t = torch.tensor(np.array(S), dtype=torch.float32)

        V = critic.net(S_t).squeeze(-1)
        closs = F.mse_loss(V / 1e4, ret_t / 1e4)
        critic_opt.zero_grad(); closs.backward(); critic_opt.step()

        with torch.no_grad():
            adv = ret_t - critic.net(S_t).squeeze(-1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        LP = torch.stack(LOGP)                     # [T, N]
        # single backward over the summed per-agent losses: each agent's
        # log-probs depend only on its own parameters, so this is identical to
        # N separate updates but avoids in-place parameter mutation mid-graph
        for o in optimizers:
            o.zero_grad()
        total_loss = sum(-(LP[:, i] * adv).mean() for i in range(N))
        total_loss.backward()
        for i in range(N):
            torch.nn.utils.clip_grad_norm_(agents[i].parameters(), 1.0)
        for o in optimizers:
            o.step()

        curve.append(ep_reward)
        if verbose and ep % max(1, episodes // 6) == 0:
            print(f"  Episode {ep:3d} | Reward: {ep_reward:9.1f}")

    for a in agents:
        a.eval()
    mediator.train_surrogate(env, agents=agents,
                             num_collections=surrogate_collections,
                             epochs=surr_epochs, verbose=verbose)

    if verbose:
        print("Applying post-training dynamic 8-bit quantization...")
    q_agents = []
    for a in agents:
        a.eval()
        q_agents.append(torch.quantization.quantize_dynamic(
            a, {nn.Linear}, dtype=torch.qint8))
    mediator.fast_surrogate.eval()
    float_surrogate = mediator.fast_surrogate
    mediator.fast_surrogate = torch.quantization.quantize_dynamic(
        mediator.fast_surrogate, {nn.Linear}, dtype=torch.qint8)
    q_after = mediator.calibrate_conformal(SmartGridEnv(N=N), q_agents)
    if q_after > 0.05:
        mediator.fast_surrogate = float_surrogate
        q_after = mediator.calibrate_conformal(SmartGridEnv(N=N), q_agents)
        if verbose:
            print(f"Quantized surrogate failed conformal admission; keeping "
                  f"float surrogate (q = {q_after:.4f}).")
    if verbose:
        print(f"Post-quantization conformal q = {mediator.conformal_q:.4f}\n")
    return mediator, q_agents, curve


# ====----------==== 8. THE CORE EVALUATION =====-----------------===============

def _pack(ep_rewards, ep_viol, violations, effs, ginis, lats, aligns, robust,
          steps, mediator=None, u0=0, m0=0, a0=0):
    out = {
        "avg_reward": float(np.mean(ep_rewards)),
        "ep_rewards": ep_rewards,
        "ep_violations": ep_viol,
        "violations_per_step": violations / steps,
        "efficiency": float(np.mean(effs) * 100),
        "alignment": float(np.mean(aligns)),
        "fairness_gini": float(np.mean(ginis)),
        "latency_ms": float(np.mean(lats)),
        "robustness": float(np.mean(robust)),
    }
    if mediator is not None:
        tot = max(1, mediator.total_mediations - m0)
        out["fast_share_eval"] = 100.0 * (mediator.fast_path_uses - u0) / tot
        out["anomaly_rate_eval"] = 100.0 * (mediator.anomaly_steps - a0) / tot
    return out


def evaluate(mediator, agents, N, episodes, attack=None, seed_offset=0,
             attack_frac=0.20, fdia_lo=20.0, fdia_hi=35.0,
             oracle_high_risk: bool = True):
    """oracle_high_risk: in all attacked evaluations of the paper the
    system is in HIGH-RISK MODE for the whole episode (is_high_risk=True), i.e.
    an operator / upstream IDS alert routes every decision to the full path.
    oracle_high_risk=False removes that alert, so the fast path is gated ONLY by
    the runtime gates (previous alignment, anomaly flags, conformal admission);
    used by run_conformal_coverage.py.  The default reproduces the paper."""
    np.random.seed(42 + seed_offset)
    torch.manual_seed(42 + seed_offset)
    env = SmartGridEnv(N=N)
    mediator.security.reset()
    u0, m0, a0 = (mediator.fast_path_uses, mediator.total_mediations,
                  mediator.anomaly_steps)
    ep_rewards, ep_viol = [], []
    effs, ginis, lats, aligns, robust = [], [], [], [], []
    violations = 0
    vs = {"viol_shortfall": 0, "viol_voltage": 0, "viol_line": 0}
    per_step_viol = np.zeros(env.T); per_step_eff = np.zeros(env.T)
    counts = np.zeros(env.T)

    for _ in range(episodes):
        s = env.reset(); ep_r = 0.0; ep_v = 0
        for t in range(env.T):
            proposals = [float(agents[i].act(s, deterministic=True)[0])
                         for i in range(N)]
            if attack == "fdia":
                proposals = apply_fdia(proposals, env, frac=attack_frac,
                                       lo=fdia_lo, hi=fdia_hi)
            elif attack == "fdia_flow":
                proposals = apply_fdia_flow(proposals, env, frac=attack_frac,
                                            lo=fdia_lo, hi=fdia_hi)
            elif attack == "byzantine":
                proposals = apply_byzantine(proposals, env)
            a, lat, align = mediator.mediate(proposals, s, env,
                                             is_high_risk=(attack is not None
                                                           and oracle_high_risk))
            s2, r, done, info = env.step(a)
            robust.append(1.0 if info["violation"] == 0 else 0.0)
            ep_r += r; ep_v += info["violation"]; violations += info["violation"]
            for k in vs:
                vs[k] += info[k]
            per_step_viol[t] += info["violation"]; per_step_eff[t] += info["efficiency"]
            counts[t] += 1
            effs.append(info["efficiency"]); ginis.append(info["gini"])
            lats.append(lat); aligns.append(align)
            s = s2
            if done:
                break
        ep_rewards.append(ep_r); ep_viol.append(ep_v)

    out = _pack(ep_rewards, ep_viol, violations, effs, ginis, lats, aligns,
                robust, episodes * env.T, mediator, u0, m0, a0)
    out["per_step_viol"] = per_step_viol / np.maximum(counts, 1)
    out["per_step_eff"] = per_step_eff / np.maximum(counts, 1)
    out.update({k: v / (episodes * env.T) for k, v in vs.items()})
    return out




class JointSafetyShield:
    """Centralised joint-action safety shield ("physics shield").

    A like-for-like, continuous-action baseline that DOES coordinate the agents
    at the network level: the joint proposal p is projected onto the feasible
    set of the linearised DC model (minimal intervention),
        min_a ||a - p||_2^2
        s.t.  a_min <= a_i <= a_max,        0 <= s_i + a_i <= s_max,
              |A (s + a)|_l            <= f_max   (line flows),
              |V_SLOPE * Th (s + a)|_i <= V_TOL   (voltage proxy),
              sum_i (s_i + a_i)       >= D_t     (adequacy).
    It is the textbook safety layer of Dalal et al. lifted from per-agent to
    joint network constraints.  The constraints are linear in a, so SLSQP is
    given their exact Jacobians.  It has no fairness term, no conformal fast
    path, no anomaly detector and no constitution; it is the physics-shielded
    MARL of Chen et al. (IEEE TSG 2023) transposed to this environment."""

    def __init__(self, a_min: float = -20.0, a_max: float = 20.0):
        self.a_min, self.a_max = a_min, a_max
        self.n_solves = 0
        self.n_success = 0

    def project(self, p: np.ndarray, env: "SmartGridEnv") -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=np.float64), -1e3, 1e3)
        s = env.supply
        N = env.N
        A, Th = env._A, env._Th
        vs = env.V_SLOPE
        fl = env.f_max
        vt = env.V_TOL
        As, Ts = A @ s, vs * (Th @ s)
        G = np.vstack([A, -A, vs * Th, -vs * Th, -np.ones((1, N))])
        h = np.concatenate([fl - As, fl + As, vt - Ts, vt + Ts,
                            [float(np.sum(s)) - env.demand]])
        lb = np.maximum(self.a_min, -s)
        ub = np.minimum(self.a_max, env.SUPPLY_CAP - s)
        x0 = np.clip(p, lb, ub)
        res = minimize(lambda a: float(np.sum((a - p) ** 2)), x0,
                       jac=lambda a: 2.0 * (a - p), method="SLSQP",
                       bounds=list(zip(lb, ub)),
                       constraints=[{"type": "ineq",
                                     "fun": lambda a: h - G @ a,
                                     "jac": lambda a: -G}],
                       options={"maxiter": 100, "ftol": 1e-9})
        self.n_solves += 1
        self.n_success += int(res.success)
        return np.clip(res.x, self.a_min, self.a_max)


def evaluate_policy_only(agents, N, episodes, mode="greedy", seed_offset=0,
                         attack=None, attack_frac=0.20, shield=None,
                         fdia_lo=20.0, fdia_hi=35.0):
    """Baselines that need no mediator.
       mode='greedy'     : raw proposals executed  (No Mediator)
       mode='projection' : proposals passed through the constitutional filter
                           only (safety-layer baseline, Dalal et al.)
       mode='shield'     : proposals projected by a
                           JointSafetyShield (joint network-level shield)"""
    np.random.seed(42 + seed_offset)
    torch.manual_seed(42 + seed_offset)
    env = SmartGridEnv(N=N)
    scorer = SafeAgentGridMediator(N, 2 * N + 1)
    ep_rewards, ep_viol = [], []
    effs, ginis, lats, aligns, robust = [], [], [], [], []
    violations = 0
    for _ in range(episodes):
        s = env.reset(); ep_r = 0.0; ep_v = 0
        for _ in range(env.T):
            t0 = time.perf_counter()
            proposals = [float(agents[i].act(s, deterministic=True)[0])
                         for i in range(N)]
            if attack == "fdia":
                proposals = apply_fdia(proposals, env, frac=attack_frac,
                                       lo=fdia_lo, hi=fdia_hi)
            elif attack == "fdia_flow":
                proposals = apply_fdia_flow(proposals, env, frac=attack_frac,
                                            lo=fdia_lo, hi=fdia_hi)
            elif attack == "byzantine":
                proposals = apply_byzantine(proposals, env)
            a = np.asarray(proposals, dtype=np.float64)
            if mode == "projection":
                a = scorer.constitution.filter(a)
            elif mode == "shield":
                a = shield.project(a, env)
            lats.append((time.perf_counter() - t0) * 1000)
            aligns.append(scorer.alignment_score(a, env))
            s, r, done, info = env.step(a)
            ep_r += r; ep_v += info["violation"]; violations += info["violation"]
            robust.append(1.0 if info["violation"] == 0 else 0.0)
            effs.append(info["efficiency"]); ginis.append(info["gini"])
            if done:
                break
        ep_rewards.append(ep_r); ep_viol.append(ep_v)
    return _pack(ep_rewards, ep_viol, violations, effs, ginis, lats, aligns,
                 robust, episodes * env.T)
     


def train_unmediated(episodes, N, use_gat=True, shared_policy=False,
                     constitutional=False, verbose=False, seed=None,
                     shield=None, lagrangian=False, cost_limit=0.0,
                     lag_lr=50.0):
    """Other techniques   trained in THIS environment, without the mediator.
       shared_policy=True + constitutional=True -> 'Single-Agent Constitutional'
       otherwise                                -> 'Independent PG MARL (CTDE critic)'
    seed   : re-seed numpy/torch before training (independent runs).
    shield : a JointSafetyShield applied IN THE TRAINING LOOP, so the
                        agents learn with the network-level shield
                        (physics-shielded MARL baseline).
               lagrangian : Lagrangian constrained PG (MAPPO-L-style dual
                        ascent): per-step cost c_t = line overload (pu) +
                        max(0, shortfall - 5 MW)/S_base, reward r_t - lam*c_t,
                        and lam <- max(0, lam + lag_lr * (mean c - cost_limit))
                        after every episode."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    state_dim = 2 * N + 1
    env = SmartGridEnv(N=N)
    if shared_policy:
        base = AgentPolicy(state_dim, use_gat=use_gat, N=N)
        agents = [base] * N
        opts = [optim.Adam(base.parameters(), lr=3e-4)]
    else:
        agents = [AgentPolicy(state_dim, use_gat=use_gat, N=N) for _ in range(N)]
        opts = [optim.Adam(a.parameters(), lr=3e-4) for a in agents]
    critic = ValueBaseline(state_dim)
    copt = optim.Adam(critic.parameters(), lr=1e-3)
    filt = Constitution()
    gamma = 0.99
    lam_c = 0.0                                   # Lagrange multiplier
    for ep in range(episodes):
        s = env.reset(); S, R, LOGP = [], [], []
        C = []                                    # per-step costs
        for _ in range(env.T):
            logps, props = [], []
            for i in range(N):
                a_i, lp = agents[i].act(s)
                props.append(float(a_i)); logps.append(lp)
            a = np.asarray(props)
            if constitutional:
                a = filt.filter(a)
            if shield is not None:
                a = shield.project(a, env)
            s2, r, done, inf = env.step(a)
            if lagrangian:
                c = (inf["line_overload"]
                     + max(0.0, inf["shortfall"] - env.SHORTFALL_TOL) / env.S_BASE)
                C.append(c)
                r = r - lam_c * c
            S.append(s.copy()); R.append(r); LOGP.append(torch.stack(logps))
            s = s2
            if done:
                break
        if lagrangian:                            # dual ascent
            lam_c = max(0.0, lam_c + lag_lr * (float(np.mean(C)) - cost_limit))
        G, rets = 0.0, []
        for r in reversed(R):
            G = r + gamma * G; rets.insert(0, G)
        ret_t = torch.tensor(rets, dtype=torch.float32)
        S_t = torch.tensor(np.array(S), dtype=torch.float32)
        V = critic.net(S_t).squeeze(-1)
        cl = F.mse_loss(V / 1e4, ret_t / 1e4)
        copt.zero_grad(); cl.backward(); copt.step()
        with torch.no_grad():
            adv = ret_t - critic.net(S_t).squeeze(-1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        LP = torch.stack(LOGP)
        if shared_policy:
            opts[0].zero_grad()
            loss = -(LP * adv.unsqueeze(1)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agents[0].parameters(), 1.0)
            opts[0].step()
        else:
            for o in opts:
                o.zero_grad()
            total_loss = sum(-(LP[:, i] * adv).mean() for i in range(N))
            total_loss.backward()
            for i in range(N):
                torch.nn.utils.clip_grad_norm_(agents[i].parameters(), 1.0)
            for o in opts:
                o.step()
    for a in agents:
        a.eval()
    return agents


# ===================== 9. THE STATISTICS DETIALS    -----=====================




def run_multi_seed(mediator, agents, N, episodes, seeds=5):
    print(f"\n=== MULTI-SEED STATISTICS ({seeds} evaluation seeds, "
          f"single training run) ===")
    res = []
    for sd in range(seeds):
        mediator.reset_telemetry()
        res.append(evaluate(mediator, agents, N, episodes, seed_offset=sd))
    keys = ["avg_reward", "violations_per_step", "efficiency", "alignment",
            "fairness_gini", "latency_ms", "robustness"]
    mean = {k: float(np.mean([r[k] for r in res])) for k in keys}
    std = {k: float(np.std([r[k] for r in res], ddof=1)) for k in keys}
    for k in keys:
        print(f"  {k:22s}: {mean[k]:10.4f} +/- {std[k]:.4f}")
    return mean, std


def run_wilcoxon(ours, baselines):
    """Test BOTH reward and violations, and report the direction."""
    print("\n=== WILCOXON SIGNED-RANK (paired per-episode) ===")
    out = {}
    for name, res in baselines.items():
        n = min(len(ours["ep_rewards"]), len(res["ep_rewards"]))
        row = {}
        for metric, key, better in [("reward", "ep_rewards", "higher"),
                                    ("violations", "ep_violations", "lower")]:
            x = np.asarray(ours[key][:n], dtype=float)
            y = np.asarray(res[key][:n], dtype=float)
            if np.allclose(x, y):
                row[metric] = (float("nan"), float("nan"), "identical")
                print(f"  {name:26s} {metric:11s}: identical, test not applicable")
                continue
            stat, p = wilcoxon(x, y)
            if better == "higher":
                direction = ("SafeAgentGrid higher" if x.mean() > y.mean()
                             else "baseline higher")
            else:
                direction = ("SafeAgentGrid lower (better)" if x.mean() < y.mean()
                             else "baseline lower")
            row[metric] = (float(stat), float(p), direction)
            print(f"  {name:26s} {metric:11s}: W={stat:6.1f}, p={p:.2e}  "
                  f"-> {direction}")
        out[name] = row
    return out


# ====######================= 10. THE ABLATION STUDY / SWEEPS =====  -----================

def run_ablation(med_full, ag_full, train_kwargs, N, eval_episodes):
    """Every row is evaluated under the SAME clean condition; a matching
    attacked column is reported alongside so the security row is comparable."""
    print("\n=== ABLATION STUDY ===")
    rows = {}

    def _both(med, ags):
        med.reset_telemetry()
        clean = evaluate(med, ags, N, eval_episodes)
        med.reset_telemetry()
        atk = evaluate(med, ags, N, eval_episodes, attack="fdia")
        return {"clean": clean, "fdia": atk}

    print("[ablation] Full SafeAgentGrid...")
    rows["Full SafeAgentGrid"] = _both(med_full, ag_full)

    print("[ablation] w/o Hybrid Fast Path...")
    med_nf = SafeAgentGridMediator(N, 2 * N + 1, fast_path_enabled=False)
    med_nf.fast_surrogate = med_full.fast_surrogate
    med_nf.conformal_q = med_full.conformal_q
    med_nf.surrogate_trained = True
    rows["w/o Hybrid Fast Path"] = _both(med_nf, ag_full)





    print("[ablation] w/o GAT Layer (trained with same budget)...")
    med_ng, ag_ng, _ = train_safeagentgrid(
        train_kwargs["episodes"], N, train_kwargs["surrogate_collections"],
        use_gat=False, verbose=False, surr_epochs=train_kwargs["surr_epochs"])
    rows["w/o GAT Layer"] = _both(med_ng, ag_ng)

    print("[ablation] w/o Security Module...")
    med_ns = SafeAgentGridMediator(N, 2 * N + 1, security_enabled=False)
    med_ns.fast_surrogate = med_full.fast_surrogate
    med_ns.conformal_q = med_full.conformal_q
    med_ns.surrogate_trained = True
    rows["w/o Security Module"] = _both(med_ns, ag_full)

    print("[ablation] w/o Mediator (greedy)...")
    rows["w/o Mediator"] = {
        "clean": evaluate_policy_only(ag_full, N, eval_episodes, mode="greedy"),
        "fdia": evaluate_policy_only(ag_full, N, eval_episodes, mode="greedy",
                                     attack="fdia")}

    hdr = (f"{'Configuration':24s} | {'Viol/step':>18s} | {'Reward':>10s} | "
           f"{'Eff %':>6s} | {'Align':>6s} | {'Lat ms':>7s}")
    print(hdr); print("-" * len(hdr))
    for name, r in rows.items():
        c, f = r["clean"], r["fdia"]
        print(f"{name:24s} | {c['violations_per_step']:.4f} / "
              f"{f['violations_per_step']:.4f} | {c['avg_reward']:10.1f} | "
              f"{c['efficiency']:6.1f} | {c['alignment']:6.3f} | "
              f"{c['latency_ms']:7.2f}")
    return rows


def run_adversarial_comparison(mediator, agents, N, episodes, normal, fdia,
                               greedy):
    print("\n=== ADVERSARIAL COMPARISON (defended vs. undefended, 20% FDIA) ===")
    rows = {"SafeAgentGrid (defended)": {"normal": normal, "fdia": fdia}}
    greedy_fdia = evaluate_policy_only(agents, N, episodes, mode="greedy",
                                       attack="fdia")
    rows["No Mediator (undefended)"] = {"normal": greedy, "fdia": greedy_fdia}
    med_ns = SafeAgentGridMediator(N, 2 * N + 1, security_enabled=False)
    med_ns.fast_surrogate = mediator.fast_surrogate
    med_ns.conformal_q = mediator.conformal_q
    med_ns.surrogate_trained = True
    med_ns.reset_telemetry(); ns_norm = evaluate(med_ns, agents, N, episodes)
    med_ns.reset_telemetry()
    ns_fdia = evaluate(med_ns, agents, N, episodes, attack="fdia")
    rows["Mediator w/o Security (undefended)"] = {"normal": ns_norm, "fdia": ns_fdia}
    for name, r in rows.items():
        n, f = r["normal"], r["fdia"]
        print(f"{name:38s} | {n['avg_reward']:9.1f}/{f['avg_reward']:9.1f} | "
              f"{n['violations_per_step']:.3f}/{f['violations_per_step']:.3f} | "
              f"{f['robustness']:.3f}")

    names = list(rows.keys())
    short = ["SafeAgentGrid\n(defended)", "No Mediator\n(undefended)",
             "Mediator w/o Security\n(undefended)"]
    ret = [100.0 * rows[k]["fdia"]["avg_reward"] /
           max(1e-9, rows[k]["normal"]["avg_reward"]) for k in names]
    vio = [rows[k]["fdia"]["violations_per_step"] for k in names]
    x = np.arange(len(names))
    fig, ax1 = plt.subplots(figsize=(9, 5.5), dpi=150)
    b1 = ax1.bar(x - 0.18, ret, width=0.36, color="tab:green")
    ax1.set_ylabel("Reward retained under attack (%)", color="tab:green")
    ax1.tick_params(axis="y", labelcolor="tab:green"); ax1.set_ylim(0, 115)
    ax2 = ax1.twinx()
    b2 = ax2.bar(x + 0.18, vio, width=0.36, color="tab:red")
    ax2.set_ylabel("Violations per step under attack", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.set_xticks(x); ax1.set_xticklabels(short)
    for b, v in zip(b1, ret):
        ax1.text(b.get_x() + b.get_width()/2, v + 1.5, f"{v:.1f}%", ha="center",
                 fontsize=9, color="tab:green")
    for b, v in zip(b2, vio):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center",
                 fontsize=9, color="tab:red")
    fig.suptitle("Defended vs. Undefended Coordination under 20% FDIA",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig("adversarial_comparison.png",
                                    bbox_inches="tight"); plt.close(fig)
    return rows


def run_attack_sweep(mediator, agents, N, episodes,
                     fracs=(0.0, 0.10, 0.20, 0.30, 0.40)):
    print("\n=== ATTACK-SEVERITY SWEEP (FDIA fraction) ===")
    rows = {}
    for f in fracs:
        mediator.reset_telemetry()
        rows[f] = evaluate(mediator, agents, N, episodes,
                           attack=None if f == 0.0 else "fdia", attack_frac=f)
        r = rows[f]
        print(f"frac={f:.2f} | Reward: {r['avg_reward']:9.1f} | "
              f"Viol: {r['violations_per_step']:.4f} | "
              f"Robust: {r['robustness']:.3f} | Lat: {r['latency_ms']:.2f} ms")
    fr = list(rows.keys())
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    ax1.plot([100*f for f in fr], [rows[f]["violations_per_step"] for f in fr],
             "o-", color="tab:red", lw=2)
    ax1.set_xlabel("FDIA corruption fraction (%)")
    ax1.set_ylabel("Violations per step", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red"); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot([100*f for f in fr], [rows[f]["robustness"] for f in fr], "s--",
             color="tab:blue", lw=2)
    ax2.set_ylabel("Robustness score", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue"); ax2.set_ylim(0.0, 1.05)
    fig.suptitle("Graceful Degradation under Increasing FDIA Severity",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig("attack_sweep.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def run_stealth_sweep(mediator, agents, N, episodes,
                      mags=((1.0, 2.0), (2.0, 5.0), (5.0, 10.0), (20.0, 35.0))):
    """Bounded / stealthy FDIA: perturbations comparable to nominal
    action variation, where the MAD detector should start to fail."""
    print("\n=== STEALTH-ATTACK SWEEP (bounded FDIA magnitude, 20% agents) ===")
    rows = {}
    for lo, hi in mags:
        mediator.reset_telemetry()
        r = evaluate(mediator, agents, N, episodes, attack="fdia",
                     attack_frac=0.20, fdia_lo=lo, fdia_hi=hi)
        rows[(lo, hi)] = r
        print(f"delta ~ U[{lo:4.1f},{hi:4.1f}] | detect: "
              f"{r['anomaly_rate_eval']:5.1f}% | Viol: "
              f"{r['violations_per_step']:.4f} | Robust: {r['robustness']:.3f} "
              f"| Reward: {r['avg_reward']:9.1f}")
    return rows


def run_fairness_sweep(mediator, agents, N, episodes, scales=(1.0, 10.0, 100.0)):
    print("\n=== FAIRNESS-REWARD SWEEP (Gini weight multiplier nu) ===")
    saved_scale, saved_fp = mediator.nu_scale, mediator.fast_path_enabled
    mediator.fast_path_enabled = False
    rows = {}
    for sc in scales:
        mediator.nu_scale = sc
        mediator.reset_telemetry()
        rows[sc] = evaluate(mediator, agents, N, episodes)
        r = rows[sc]
        print(f"nu x{sc:6.1f} | Reward: {r['avg_reward']:9.1f} | Gini: "
              f"{r['fairness_gini']:.4f} | Viol: {r['violations_per_step']:.4f}")
    mediator.nu_scale, mediator.fast_path_enabled = saved_scale, saved_fp
    return rows


def run_fastpath_sweep(mediator, agents, N, episodes,
                       gates=(0.0, 0.90, 0.95, 0.99, 2.0)):
    print("\n=== FAST-PATH GATE SWEEP (speed-fairness-safety dial) ===")
    saved = mediator.fast_gate
    rows = {}
    for g in gates:
        mediator.fast_gate = g
        mediator.reset_telemetry()
        rows[g] = evaluate(mediator, agents, N, episodes)
        rows[g]["fast_share"] = rows[g]["fast_share_eval"]
        r = rows[g]
        print(f"gate={g:4.2f} | fast: {r['fast_share']:5.1f}% | Reward: "
              f"{r['avg_reward']:9.1f} | Gini: {r['fairness_gini']:.4f} | "
              f"Viol: {r['violations_per_step']:.4f} | "
              f"Lat: {r['latency_ms']:.2f} ms")
    mediator.fast_gate = saved
    xs = [rows[g]["fast_share"] for g in gates]
    order = np.argsort(xs); xs = [xs[i] for i in order]
    gl = [gates[i] for i in order]
    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=150)
    ax1.plot(xs, [rows[g]["fairness_gini"] for g in gl], "o-",
             color="tab:purple", lw=2, label="Fairness (Gini)")
    ax1.plot(xs, [rows[g]["violations_per_step"] for g in gl], "d-",
             color="tab:red", lw=2, label="Violations/step")
    ax1.set_xlabel("Fast-path share of decisions (%)")
    ax1.set_ylabel("Gini / violations per step"); ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(xs, [rows[g]["latency_ms"] for g in gl], "s--", color="tab:orange",
             lw=2, label="Latency (ms)")
    ax2.set_ylabel("Latency (ms/step)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    l1, b1 = ax1.get_legend_handles_labels(); l2, b2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, b1 + b2, loc="center left", fontsize=9)
    fig.suptitle("Speed-Fairness-Safety Dial: Fast-Path Admission Gate",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig("fastpath_tradeoff.png", bbox_inches="tight")
    plt.close(fig)
    return rows


def run_scalability(train_kwargs, eval_episodes, Ns=(4, 8, 10, 16),
                    precomputed=None):
    print("\n=== SCALABILITY SWEEP ===")
    rows = {}
    for n in Ns:
        if precomputed and n in precomputed:
            rows[n] = precomputed[n]
        else:
            med, ags, _ = train_safeagentgrid(
                train_kwargs["episodes"], n,
                train_kwargs["surrogate_collections"], verbose=False,
                surr_epochs=train_kwargs["surr_epochs"])
            med.reset_telemetry()
            rows[n] = evaluate(med, ags, n, eval_episodes)
        r = rows[n]
        print(f"N={n:2d} | Reward: {r['avg_reward']:9.1f} | "
              f"Reward/agent: {r['avg_reward']/n:8.1f} | "
              f"Viol: {r['violations_per_step']:.4f} | "
              f"Eff: {r['efficiency']:.1f}% | Lat: {r['latency_ms']:.2f} ms")
    return rows


# ==============#####____=== 11. THE FIGURES PLUS      LATEX EXPORT =====================

def generate_results_figure(mediator, agents, N, episodes):
    mediator.reset_telemetry()
    res = evaluate(mediator, agents, N, episodes)
    np.random.seed(42); torch.manual_seed(42)
    env = SmartGridEnv(N=N); s = env.reset()
    cum, sup, dem, allo = [], [], [], []
    tot = 0.0
    for _ in range(env.T):
        props = [float(agents[i].act(s, deterministic=True)[0]) for i in range(N)]
        a, _, _ = mediator.mediate(props, s, env)
        s, r, done, info = env.step(a)
        tot += r; cum.append(tot)
        sup.append(float(np.sum(env.supply))); dem.append(env.demand)
        allo.append(min(float(np.sum(env.supply)), env.demand))
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), dpi=150)
    ax[0, 0].plot(cum, color="tab:blue", lw=2)
    ax[0, 0].axhline(cum[-1], ls="--", color="tab:red",
                     label=f"Final: {cum[-1]:.1f}")
    ax[0, 0].set_title("Cumulative Reward (final evaluation episode)")
    ax[0, 0].set_xlabel("Step"); ax[0, 0].set_ylabel("Cumulative Reward")
    ax[0, 0].legend()
    ax[0, 1].plot(res["per_step_viol"], color="tab:red", lw=1)
    mv = float(np.mean(res["per_step_viol"]))
    ax[0, 1].axhline(mv, ls="--", color="tab:blue", label=f"Mean: {mv:.4f}")
    ax[0, 1].set_title("Average Violations per Step")
    ax[0, 1].set_xlabel("Step"); ax[0, 1].set_ylabel("Violations"); ax[0, 1].legend()
    ax[1, 0].plot(100 * res["per_step_eff"], color="tab:green", lw=1)
    me = float(np.mean(res["per_step_eff"]) * 100)
    ax[1, 0].axhline(me, ls="--", color="tab:red", label=f"Mean: {me:.2f}%")
    ax[1, 0].set_title("Grid Efficiency (%)"); ax[1, 0].set_xlabel("Step")
    ax[1, 0].set_ylabel("Efficiency (%)"); ax[1, 0].legend()
    ax[1, 1].plot(sup, label="Total Supply", color="tab:blue")
    ax[1, 1].plot(dem, label="Demand", color="tab:orange")
    ax[1, 1].plot(allo, "--", label="Allocated", color="tab:green")
    ax[1, 1].set_title("Supply vs Demand (last episode)")
    ax[1, 1].set_xlabel("Step"); ax[1, 1].set_ylabel("Power (MW)"); ax[1, 1].legend()
    fig.suptitle(f"SafeAgentGrid Simulation Results (N={N}, {env.T}-step episodes)",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig("Results.png", bbox_inches="tight"); plt.close(fig)
    return {"final_cum_reward": cum[-1], "mean_viol": mv, "mean_eff": me}


def generate_comparison_bar(rows: Dict[str, Dict]):
    names = list(rows.keys())
    eff = [rows[k]["efficiency"] for k in names]
    ali = [100 * rows[k]["alignment"] for k in names]
    saf = [100 * (1 - rows[k]["violations_per_step"]) for k in names]
    rew = [rows[k]["avg_reward"] for k in names]
    x = np.arange(len(names)); w = 0.2
    fig, ax1 = plt.subplots(figsize=(13, 6.5), dpi=150)
    ax1.bar(x - 1.5*w, eff, w, label="Efficiency (%)", color="tab:green")
    ax1.bar(x - 0.5*w, ali, w, label="Alignment (%)", color="orange")
    ax1.bar(x + 0.5*w, saf, w, label="Safety Score (%)", color="tab:red")
    ax1.set_ylabel("Percentage / Scaled Score")
    ax2 = ax1.twinx()
    ax2.bar(x + 1.5*w, rew, w, label="Avg. Reward", color="tab:blue")
    ax2.set_ylabel("Avg. Reward")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace(" (", "\n(") for n in names], fontsize=8)
    l1, b1 = ax1.get_legend_handles_labels(); l2, b2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, b1 + b2, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle("Comparison with Baselines Implemented in This Environment",
                 fontweight="bold")
    fig.tight_layout(); fig.savefig("comparison_bar_improved.png",
                                    bbox_inches="tight"); plt.close(fig)


def export_latex(path, main, ms_mean, ms_std, ablation, scal, sweep, adv,
                 fp, fair, stealth, lat_fast, lat_full, q):
    L = []
    A = L.append
    A("% ================ Table 6: main results (N=10) =================")
    A("% Method & Reward & Viol/step (norm) & Viol/step (20% FDIA) & Eff (%) "
      "& Align & Gini & Latency (ms/step)")
    for name, r in main.items():
        fd = r.get("fdia_viol")
        fd = f"{fd:.3f}" if fd is not None else "--"
        A(f"{name} & {r['avg_reward']:.1f} & {r['violations_per_step']:.3f} & "
          f"{fd} & {r['efficiency']:.1f} & {r['alignment']:.2f} & "
          f"{r['fairness_gini']:.3f} & {r['latency_ms']:.2f} \\\\")
    A("")
    A("% ---- multi-seed mean +/- std (5 evaluation seeds, single training run)")
    for k in ms_mean:
        A(f"% {k}: {ms_mean[k]:.4f} +/- {ms_std[k]:.4f}")
    A("")
    A("% ================ Table 9: ablation (clean / 20% FDIA) ================")
    A("% Config & Viol/step clean & Viol/step FDIA & Reward & Eff & Align & Lat")
    for name, r in ablation.items():
        c, f = r["clean"], r["fdia"]
        A(f"{name} & {c['violations_per_step']:.3f} & "
          f"{f['violations_per_step']:.3f} & {c['avg_reward']:.1f} & "
          f"{c['efficiency']:.1f} & {c['alignment']:.2f} & "
          f"{c['latency_ms']:.2f} \\\\")
    A("")
    A("% ================ Table 10: scalability ===============================")
    A("% N & Reward & Reward/agent & Viol/step & Eff (%) & Latency (ms)")
    for n, r in scal.items():
        A(f"{n} & {r['avg_reward']:.1f} & {r['avg_reward']/n:.1f} & "
          f"{r['violations_per_step']:.3f} & {r['efficiency']:.1f} & "
          f"{r['latency_ms']:.2f} \\\\")
    A("")
    A("% ================ Table 11: FDIA severity sweep =======================")
    for f, r in sweep.items():
        A(f"{100*f:.0f}\\% & {r['avg_reward']:.1f} & "
          f"{r['violations_per_step']:.3f} & {r['robustness']:.3f} & "
          f"{r['latency_ms']:.2f} \\\\")
    A("")
    A("% ================ Table 12: defended vs undefended ====================")
    for name, r in adv.items():
        n, f = r["normal"], r["fdia"]
        A(f"{name} & {n['avg_reward']:.1f} & {f['avg_reward']:.1f} & "
          f"{n['violations_per_step']:.3f} & {f['violations_per_step']:.3f} & "
          f"{f['robustness']:.3f} \\\\")
    A("")
    A("% ================ Table 13: fast-path gate sweep ======================")
    for g, r in fp.items():
        A(f"{g:g} & {r['fast_share']:.1f}\\% & {r['avg_reward']:.1f} & "
          f"{r['fairness_gini']:.3f} & {r['violations_per_step']:.3f} & "
          f"{r['latency_ms']:.2f} \\\\")
    A("")
    A("% ================ Table 14: stealth-attack sweep ======================")
    A("% delta range & detection rate (%) & Viol/step & Robustness & Reward")
    for (lo, hi), r in stealth.items():
        A(f"$\\delta\\sim U[{lo:g},{hi:g}]$ & {r['anomaly_rate_eval']:.1f} & "
          f"{r['violations_per_step']:.3f} & {r['robustness']:.3f} & "
          f"{r['avg_reward']:.1f} \\\\")
    A("")
    A("% ================ fairness sweep ======================================")
    for sc, r in fair.items():
        A(f"$\\nu\\times{sc:g}$ & {r['avg_reward']:.1f} & "
          f"{r['fairness_gini']:.4f} & {r['violations_per_step']:.3f} \\\\")
    A("")
    A(f"% latency decomposition: fast {lat_fast:.3f} ms, full {lat_full:.3f} ms, "
      f"conformal q = {q:.4f}")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"LaTeX rows written to {path}")


# ===================== 12.THE    MAIN =====#################================


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--medium", action="store_true")
    args, _ = ap.parse_known_args()

    if args.quick:
        cfg = dict(train_eps=5, N=10, surr=6, eval_eps=3, seeds=2,
                   scal_Ns=(4, 10), fig_eps=3, surr_epochs=40)
    elif args.medium:
        cfg = dict(train_eps=30, N=10, surr=40, eval_eps=10, seeds=5,
                   scal_Ns=(4, 8, 10, 16), fig_eps=10, surr_epochs=200)
    else:
        cfg = dict(train_eps=300, N=10, surr=100, eval_eps=50, seeds=10,
                   scal_Ns=(4, 8, 10, 16), fig_eps=50, surr_epochs=400)

    print("=" * 70)
    print("SAFEAGENTGRID - FULL PIPELINE")
    print("=" * 70)

    train_kwargs = dict(episodes=cfg["train_eps"], N=cfg["N"],
                        surrogate_collections=cfg["surr"],
                        surr_epochs=cfg["surr_epochs"])
    N = cfg["N"]
    mediator, agents, curve = train_safeagentgrid(**train_kwargs)
    print(f"Training reward curve: first={curve[0]:.1f} last={curve[-1]:.1f} "
          f"mean(first 5)={np.mean(curve[:5]):.1f} "
          f"mean(last 5)={np.mean(curve[-5:]):.1f}")

    mediator.reset_telemetry()
    print("\nEvaluation: normal conditions")
    normal = evaluate(mediator, agents, N, cfg["eval_eps"])
    print({k: round(v, 4) for k, v in normal.items() if isinstance(v, (int, float))})
    lat_fast = float(np.mean(mediator.fast_latencies)) if mediator.fast_latencies else float("nan")
    lat_full = float(np.mean(mediator.full_latencies)) if mediator.full_latencies else float("nan")
    print(f"Latency decomposition (this evaluation only): fast {lat_fast:.3f} ms | "
          f"full {lat_full:.3f} ms | conformal q = {mediator.conformal_q:.4f}")


    mediator.reset_telemetry()
    print("\nEvaluation: 20% FDIA")
    fdia = evaluate(mediator, agents, N, cfg["eval_eps"], attack="fdia")
    print({k: round(v, 4) for k, v in fdia.items() if isinstance(v, (int, float))})

    mediator.reset_telemetry()
    print("\nEvaluation: Byzantine")
    byz = evaluate(mediator, agents, N, cfg["eval_eps"], attack="byzantine")
    print({k: round(v, 4) for k, v in byz.items() if isinstance(v, (int, float))})

    print("\nBaselines trained in this environment...")
    greedy = evaluate_policy_only(agents, N, cfg["eval_eps"], mode="greedy")
    projonly = evaluate_policy_only(agents, N, cfg["eval_eps"], mode="projection")
    ag_ind = train_unmediated(cfg["train_eps"], N, shared_policy=False,
                              constitutional=False)
    ind = evaluate_policy_only(ag_ind, N, cfg["eval_eps"], mode="greedy")
    ag_sac = train_unmediated(cfg["train_eps"], N, shared_policy=True,
                              constitutional=True)
    sac = evaluate_policy_only(ag_sac, N, cfg["eval_eps"], mode="projection")

    greedy_f = evaluate_policy_only(agents, N, cfg["eval_eps"], mode="greedy",
                                    attack="fdia")
    proj_f = evaluate_policy_only(agents, N, cfg["eval_eps"], mode="projection",
                                  attack="fdia")
    ind_f = evaluate_policy_only(ag_ind, N, cfg["eval_eps"], mode="greedy",
                                 attack="fdia")
    sac_f = evaluate_policy_only(ag_sac, N, cfg["eval_eps"], mode="projection",
                                 attack="fdia")

    main = {
        "SafeAgentGrid (Ours)": dict(normal, fdia_viol=fdia["violations_per_step"]),
        "No Mediator (greedy)": dict(greedy, fdia_viol=greedy_f["violations_per_step"]),
        "Projection-only safety layer": dict(projonly, fdia_viol=proj_f["violations_per_step"]),
        "Independent PG MARL (CTDE critic)": dict(ind, fdia_viol=ind_f["violations_per_step"]),
        "Single-Agent Constitutional": dict(sac, fdia_viol=sac_f["violations_per_step"]),
    }
    print(f"\n{'Method':36s} | {'Reward':>10s} | {'V/step':>7s} | "
          f"{'Eff%':>6s} | {'Align':>6s} | {'Gini':>6s} | {'Lat':>6s}")
    for k, r in main.items():
        print(f"{k:36s} | {r['avg_reward']:10.1f} | "
              f"{r['violations_per_step']:7.3f} | {r['efficiency']:6.1f} | "
              f"{r['alignment']:6.3f} | {r['fairness_gini']:6.3f} | "
              f"{r['latency_ms']:6.2f}")

    adv = run_adversarial_comparison(mediator, agents, N, cfg["eval_eps"],
                                     normal, fdia, greedy)
    fair = run_fairness_sweep(mediator, agents, N, cfg["eval_eps"])
    fp = run_fastpath_sweep(mediator, agents, N, cfg["eval_eps"])
    ms_mean, ms_std = run_multi_seed(mediator, agents, N, cfg["eval_eps"],
                                     seeds=cfg["seeds"])
    wil = run_wilcoxon(normal, {"No Mediator": greedy,
                                "Independent PG MARL": ind,
                                "Single-Agent Constitutional": sac})
    sweep = run_attack_sweep(mediator, agents, N, cfg["eval_eps"])
    stealth = run_stealth_sweep(mediator, agents, N, cfg["eval_eps"])
    ablation = run_ablation(mediator, agents, train_kwargs, N, cfg["eval_eps"])
    scal = run_scalability(train_kwargs, cfg["eval_eps"], Ns=cfg["scal_Ns"],
                           precomputed={N: normal})

    print("\nGenerating figures...")
    fig_stats = generate_results_figure(mediator, agents, N, cfg["fig_eps"])
    generate_comparison_bar(main)
    export_latex("latex_table_rows_v2.txt", main, ms_mean, ms_std, ablation,
                 scal, sweep, adv, fp, fair, stealth, lat_fast, lat_full,
                 mediator.conformal_q)

    def clean(d):
        return {k: v for k, v in d.items() if isinstance(v, (int, float))}
    with open("run_summary_v2.json", "w") as fh:
        json.dump({
            "training_curve": curve,
            "main": {k: clean(v) for k, v in main.items()},
            "byzantine": clean(byz),
            "multi_seed_mean": ms_mean, "multi_seed_std": ms_std,
            "wilcoxon": {k: {m: list(v) for m, v in r.items()}
                         for k, r in wil.items()},
            "attack_sweep": {str(k): clean(v) for k, v in sweep.items()},
            "stealth_sweep": {f"{lo}-{hi}": clean(v)
                              for (lo, hi), v in stealth.items()},
            "adversarial": {k: {m: clean(v) for m, v in r.items()}
                            for k, r in adv.items()},
            "ablation": {k: {m: clean(v) for m, v in r.items()}
                         for k, r in ablation.items()},
            "scalability": {str(k): clean(v) for k, v in scal.items()},
            "fastpath_sweep": {str(k): clean(v) for k, v in fp.items()},
            "fairness_sweep": {str(k): clean(v) for k, v in fair.items()},
            "latency_fast_ms": lat_fast, "latency_full_ms": lat_full,
            "conformal_q": mediator.conformal_q,
            "figure_stats": fig_stats}, fh, indent=2)




    print("\nDONE.")








