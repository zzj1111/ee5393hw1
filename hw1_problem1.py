"""
EE 5393 - Homework 1, Problem 1: Analyzing Chemical Reaction Networks
=====================================================================

Reactions:
  R1: 2X1 + X2  -> 4X3,  k1 = 1
  R2: X1 + 2X3  -> 3X2,  k2 = 2
  R3: X2 + X3   -> 2X1,  k3 = 3

AI Disclosure: This code was entirely generated with the assistance of Claude (AI),
prompted to implement a Gillespie-style stochastic simulation and exact enumeration
based on the problem specification and the reference code (aleae simulator by M. Riedel).
"""

import numpy as np
from math import comb
from collections import defaultdict
import time

# ============================================================
# Core: Propensities and state transitions
# ============================================================

def compute_propensities(x1, x2, x3):
    """
    Propensity for each reaction using the discrete stochastic formulation:
      a_i = k_i * product of C(x_j, n_j) for each reactant j with stoich n_j.

      a1 = k1 * C(x1,2) * C(x2,1) = 1 * [x1*(x1-1)/2] * x2
      a2 = k2 * C(x1,1) * C(x3,2) = 2 * x1 * [x3*(x3-1)/2] = x1*x3*(x3-1)
      a3 = k3 * C(x2,1) * C(x3,1) = 3 * x2 * x3
    """
    a1 = 1.0 * comb(x1, 2) * comb(x2, 1)
    a2 = 2.0 * comb(x1, 1) * comb(x3, 2)
    a3 = 3.0 * comb(x2, 1) * comb(x3, 1)
    return a1, a2, a3


def fire_reaction(x1, x2, x3, reaction):
    """Apply the stoichiometric change for the chosen reaction."""
    if reaction == 0:    # R1: 2X1 + X2 -> 4X3
        return x1 - 2, x2 - 1, x3 + 4
    elif reaction == 1:  # R2: X1 + 2X3 -> 3X2
        return x1 - 1, x2 + 3, x3 - 2
    else:                # R3: X2 + X3 -> 2X1
        return x1 + 2, x2 - 1, x3 - 1


def gillespie_step(x1, x2, x3, rng):
    """One step of Gillespie's algorithm (discrete, no time)."""
    a1, a2, a3 = compute_propensities(x1, x2, x3)
    A = a1 + a2 + a3
    if A == 0:
        return None
    r = rng.random() * A
    if r < a1:
        return fire_reaction(x1, x2, x3, 0)
    elif r < a1 + a2:
        return fire_reaction(x1, x2, x3, 1)
    else:
        return fire_reaction(x1, x2, x3, 2)


# ============================================================
# Part (a): Estimate Pr(C1), Pr(C2), Pr(C3) via simulation
# ============================================================

def simulate_part_a(num_trials=10000, seed=42):
    """
    From S = [110, 26, 55], simulate until an outcome is reached:
      C1: x1 >= 150
      C2: x2 <  10
      C3: x3 > 100
    The simulation halts as soon as any condition is met (absorbing barriers).
    """
    rng = np.random.default_rng(seed)

    c1_count = 0
    c2_count = 0
    c3_count = 0

    print("=" * 65)
    print("Part (a): Stochastic Simulation for Pr(C1), Pr(C2), Pr(C3)")
    print("  Initial state: S = [110, 26, 55]")
    print("  Outcomes:")
    print("    C1: x1 >= 150")
    print("    C2: x2 <  10")
    print("    C3: x3 >  100")
    print(f"  Trials: {num_trials}")
    print("=" * 65)

    t0 = time.time()

    for _ in range(num_trials):
        x1, x2, x3 = 110, 26, 55

        while True:
            result = gillespie_step(x1, x2, x3, rng)
            if result is None:
                break
            x1, x2, x3 = result

            done = False
            if x1 >= 150:
                c1_count += 1; done = True
            if x2 < 10:
                c2_count += 1; done = True
            if x3 > 100:
                c3_count += 1; done = True
            if done:
                break

    elapsed = time.time() - t0

    print(f"\n  Results ({elapsed:.1f}s):")
    print(f"    Pr(C1) = {c1_count:6d} / {num_trials} = {c1_count/num_trials:.6f}")
    print(f"    Pr(C2) = {c2_count:6d} / {num_trials} = {c2_count/num_trials:.6f}")
    print(f"    Pr(C3) = {c3_count:6d} / {num_trials} = {c3_count/num_trials:.6f}")
    print()

    return c1_count / num_trials, c2_count / num_trials, c3_count / num_trials


# ============================================================
# Part (b): Exact computation of mean/variance after 7 steps
# ============================================================

def exact_part_b():
    """
    From S = [9, 8, 7], enumerate all reachable states after exactly 7 steps.
    Only 7 steps and 3 reactions => state space stays small (22 states).
    Compute exact mean and variance for X1, X2, X3.
    """
    print("=" * 65)
    print("Part (b): Exact Enumeration — Mean & Variance after 7 steps")
    print("  Initial state: S = [9, 8, 7]")
    print("=" * 65)

    # state_probs: {(x1, x2, x3): probability}
    state_probs = {(9, 8, 7): 1.0}

    for step in range(7):
        next_probs = defaultdict(float)
        for (x1, x2, x3), prob in state_probs.items():
            a1, a2, a3 = compute_propensities(x1, x2, x3)
            A = a1 + a2 + a3
            if A == 0:
                next_probs[(x1, x2, x3)] += prob  # absorbing state
                continue
            for r, a in enumerate([a1, a2, a3]):
                if a > 0:
                    ns = fire_reaction(x1, x2, x3, r)
                    next_probs[ns] += prob * (a / A)
        state_probs = dict(next_probs)
        print(f"  After step {step + 1}: {len(state_probs)} distinct states")

    print()

    # Compute mean and variance for each species
    results = {}
    for idx, name in enumerate(["X1", "X2", "X3"]):
        vals_probs = [(state[idx], prob) for state, prob in state_probs.items()]
        mean = sum(v * p for v, p in vals_probs)
        e_x2 = sum(v**2 * p for v, p in vals_probs)
        var = e_x2 - mean**2
        results[name] = (mean, var)
        print(f"  {name}: mean = {mean:.6f},  variance = {var:.6f}")

    # Sanity check: total probability
    total = sum(state_probs.values())
    print(f"\n  Sanity check — total probability: {total:.10f}")

    # Show X1 distribution
    print("\n  Full probability distributions after 7 steps:")
    for idx, name in enumerate(["X1", "X2", "X3"]):
        dist = defaultdict(float)
        for state, prob in state_probs.items():
            dist[state[idx]] += prob
        print(f"\n  {name}:")
        for v in sorted(dist.keys()):
            print(f"    P({name} = {v:3d}) = {dist[v]:.6f}")

    print()
    return results


# ============================================================
# Part (b) verification: Monte Carlo simulation
# ============================================================

def simulate_part_b(num_trials=200000, seed=42):
    """Monte Carlo verification of part (b)."""
    rng = np.random.default_rng(seed)

    x1_vals = np.zeros(num_trials, dtype=np.int64)
    x2_vals = np.zeros(num_trials, dtype=np.int64)
    x3_vals = np.zeros(num_trials, dtype=np.int64)

    print("=" * 65)
    print(f"Part (b) Verification: Monte Carlo ({num_trials} trials)")
    print("=" * 65)

    t0 = time.time()
    for t in range(num_trials):
        x1, x2, x3 = 9, 8, 7
        for _ in range(7):
            result = gillespie_step(x1, x2, x3, rng)
            if result is None:
                break
            x1, x2, x3 = result
        x1_vals[t], x2_vals[t], x3_vals[t] = x1, x2, x3
    elapsed = time.time() - t0

    print(f"\n  Results ({elapsed:.1f}s):")
    for name, vals in [("X1", x1_vals), ("X2", x2_vals), ("X3", x3_vals)]:
        print(f"  {name}: mean = {np.mean(vals):.6f},  variance = {np.var(vals):.6f}")
    print()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    simulate_part_a(num_trials=10000)
    exact_part_b()
    simulate_part_b(num_trials=200000)
