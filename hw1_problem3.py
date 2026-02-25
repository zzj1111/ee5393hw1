"""
EE 5393 - Homework 1, Problem 3: Synthesizing Chemical Reaction Networks
========================================================================

General-purpose Gillespie stochastic simulator for CRNs, plus two
network designs:
  (a) Z = X0 * log2(Y0)
  (b) Y = 2^(log2(X0))

AI Disclosure: This code was generated with the assistance of Claude (AI).
The CRN designs, rate constant choices, and simulation code were all
produced by Claude based on the problem specification and the given
exponentiation/logarithm module templates.
"""

import numpy as np
from math import comb
import time

# ============================================================
# General CRN Gillespie Simulator
# ============================================================

class CRN:
    def __init__(self, species_init, reactions, name="CRN"):
        self.name = name
        self.species_names = sorted(species_init.keys())
        self.species_idx = {s: i for i, s in enumerate(self.species_names)}
        self.n_species = len(self.species_names)
        self.init_state = np.array([species_init[s] for s in self.species_names], dtype=np.int64)
        self.n_reactions = len(reactions)
        self.rates = np.array([r[2] for r in reactions], dtype=np.float64)
        self.reactants = []
        self.deltas = np.zeros((self.n_reactions, self.n_species), dtype=np.int64)
        for i, (reac, prod, rate) in enumerate(reactions):
            react_list = []
            for sp, n in reac.items():
                idx = self.species_idx[sp]
                react_list.append((idx, n))
                self.deltas[i, idx] -= n
            for sp, n in prod.items():
                idx = self.species_idx[sp]
                self.deltas[i, idx] += n
            self.reactants.append(react_list)

    def propensity(self, state, i):
        a = self.rates[i]
        for (sp_idx, stoich) in self.reactants[i]:
            a *= comb(int(state[sp_idx]), stoich)
            if a == 0:
                return 0.0
        return a

    def simulate(self, max_steps=1000000, max_time=None, rng=None, track=None):
        if rng is None:
            rng = np.random.default_rng()
        state = self.init_state.copy()
        t = 0.0
        if track:
            track_idx = [self.species_idx[s] for s in track]
            traj_t = [0.0]
            traj_s = [[int(state[i]) for i in track_idx]]
        for step in range(max_steps):
            props = np.array([self.propensity(state, i) for i in range(self.n_reactions)])
            A = props.sum()
            if A == 0:
                break
            tau = -np.log(rng.random()) / A
            t += tau
            if max_time is not None and t > max_time:
                break
            r = rng.random() * A
            cumsum = 0.0
            chosen = self.n_reactions - 1
            for i in range(self.n_reactions):
                cumsum += props[i]
                if cumsum >= r:
                    chosen = i
                    break
            state += self.deltas[chosen]
            if track:
                traj_t.append(t)
                traj_s.append([int(state[i]) for i in track_idx])
        final = {s: int(state[self.species_idx[s]]) for s in self.species_names}
        if track:
            return final, (traj_t, traj_s, track)
        return final


# ============================================================
# Rate Design Philosophy
# ============================================================
# 
# Within each module, four rate tiers enforce sequential execution:
#
#   FASTER (100000) >> FAST (1000) >> MEDIUM >> SLOW
#
# - FASTER: bulk processing   (a+2y -> c+y'+a, d+x -> d+x'+z', etc.)
# - FAST:   catalyst cleanup  (a -> empty, d -> empty, e -> empty)
# - MEDIUM: species restore   (y' -> y, x' -> x, z' -> z)
# - SLOW:   round trigger     (b -> a+b, w -> d, w -> e)
#
# Key constraint: FAST >> MEDIUM * (max molecules) so the catalyst
# dies BEFORE any restored molecules re-enter a fast reaction.
#
# Between modules, the second module's SLOW rate << first module's
# MEDIUM rate, ensuring Module 1 finishes before Module 2 consumes
# its output.
#
# Log module:   SLOW=0.1,  MEDIUM=10,  FAST=1000,  FASTER=100000
# Mult module:  SLOW=0.001, MEDIUM=0.5, FAST=1000, FASTER=100000
# Exp module:   SLOW=0.001, MEDIUM=0.5, FAST=1000, FASTER=100000
#

R_FASTER  = 100000.0
R_FAST    = 1000.0

# Log module rates
LOG_SLOW   = 0.1
LOG_MEDIUM = 10.0

# Second module (mult or exp) rates
MOD2_SLOW   = 0.001
MOD2_MEDIUM = 0.5


# ============================================================
# Network 1: Z = X0 * log2(Y0)
# ============================================================

def make_crn_x_times_log2y(x0, y0):
    """
    Z = X0 * log2(Y0)

    Module 1 -- Logarithm (W = log2(Y)):
      Pseudocode: While Y>1: Y=Y/2; W=W+1

    Module 2 -- Multiplication (Z = X * W):
      Pseudocode: ForEach w: Z += X
    """
    species = {
        'x': x0, 'y': y0, 'z': 0,
        'b': 1, 'a': 0, 'c': 0, 'yp': 0, 'w': 0,
        'd': 0, 'xp': 0, 'zp': 0,
    }
    reactions = [
        # --- Log module ---
        ({'b': 1},             {'a': 1, 'b': 1},              LOG_SLOW),     # R0
        ({'a': 1, 'y': 2},    {'c': 1, 'yp': 1, 'a': 1},     R_FASTER),     # R1
        ({'c': 2},             {'c': 1},                       R_FASTER),     # R2
        ({'a': 1},             {},                              R_FAST),       # R3
        ({'yp': 1},            {'y': 1},                        LOG_MEDIUM),   # R4
        ({'c': 1},             {'w': 1},                        LOG_MEDIUM),   # R5
        # --- Mult module ---
        ({'w': 1},             {'d': 1},                        MOD2_SLOW),    # R6
        ({'d': 1, 'x': 1},    {'d': 1, 'xp': 1, 'zp': 1},    R_FASTER),     # R7
        ({'d': 1},             {},                              R_FAST),       # R8
        ({'xp': 1},            {'x': 1},                        MOD2_MEDIUM),  # R9
        ({'zp': 1},            {'z': 1},                        MOD2_MEDIUM),  # R10
    ]
    return CRN(species, reactions, name=f"Z = {x0} * log2({y0})")


# ============================================================
# Network 2: Y = 2^(log2(X0))
# ============================================================

def make_crn_2_to_log2x(x0):
    """
    Y = 2^(log2(X0))

    Module 1 -- Logarithm (W = log2(X)):
      Pseudocode: While X>1: X=X/2; W=W+1

    Module 2 -- Exponentiation (Y = 2^W):
      Pseudocode: Y=1; ForEach w: Y=2*Y
    """
    species = {
        'x': x0, 'y': 1,
        'b': 1, 'a': 0, 'c': 0, 'xp': 0, 'w': 0,
        'e': 0, 'yp': 0,
    }
    reactions = [
        # --- Log module ---
        ({'b': 1},             {'a': 1, 'b': 1},              LOG_SLOW),
        ({'a': 1, 'x': 2},    {'c': 1, 'xp': 1, 'a': 1},     R_FASTER),
        ({'c': 2},             {'c': 1},                       R_FASTER),
        ({'a': 1},             {},                              R_FAST),
        ({'xp': 1},            {'x': 1},                        LOG_MEDIUM),
        ({'c': 1},             {'w': 1},                        LOG_MEDIUM),
        # --- Exp module ---
        ({'w': 1},             {'e': 1},                        MOD2_SLOW),
        ({'e': 1, 'y': 1},    {'e': 1, 'yp': 2},              R_FASTER),
        ({'e': 1},             {},                              R_FAST),
        ({'yp': 1},            {'y': 1},                        MOD2_MEDIUM),
    ]
    return CRN(species, reactions, name=f"Y = 2^(log2({x0}))")


# ============================================================
# Testing
# ============================================================

def test_network1(num_trials=30):
    print("=" * 70)
    print("Network 1:  Z_inf = X_0 * log2(Y_0)")
    print("=" * 70)
    test_cases = [
        (3, 8),    # 3*3=9
        (5, 4),    # 5*2=10
        (4, 16),   # 4*4=16
        (7, 32),   # 7*5=35
        (10, 8),   # 10*3=30
    ]
    print(f"\n{'X0':>4} {'Y0':>4} {'Expected':>10} {'Mean Z':>10} {'Median Z':>10} {'Trials':>7}")
    print("-" * 55)
    for x0, y0 in test_cases:
        expected = x0 * int(np.log2(y0))
        z_vals = []
        for trial in range(num_trials):
            crn = make_crn_x_times_log2y(x0, y0)
            rng = np.random.default_rng(trial * 1000 + x0 * 100 + y0)
            final = crn.simulate(max_steps=1000000, max_time=50000, rng=rng)
            z_vals.append(final['z'])
        mean_z = np.mean(z_vals)
        med_z = np.median(z_vals)
        print(f"{x0:>4} {y0:>4} {expected:>10} {mean_z:>10.1f} {med_z:>10.0f} {num_trials:>7}")
    print()


def test_network2(num_trials=30):
    print("=" * 70)
    print("Network 2:  Y_inf = 2^(log2(X_0))   [should equal X_0]")
    print("=" * 70)
    test_cases = [4, 8, 16, 32]
    print(f"\n{'X0':>4} {'Expected':>10} {'Mean Y':>10} {'Median Y':>10} {'Trials':>7}")
    print("-" * 50)
    for x0 in test_cases:
        y_vals = []
        for trial in range(num_trials):
            crn = make_crn_2_to_log2x(x0)
            rng = np.random.default_rng(trial * 1000 + x0)
            final = crn.simulate(max_steps=1000000, max_time=50000, rng=rng)
            y_vals.append(final['y'])
        mean_y = np.mean(y_vals)
        med_y = np.median(y_vals)
        print(f"{x0:>4} {x0:>10} {mean_y:>10.1f} {med_y:>10.0f} {num_trials:>7}")
    print()


def demo_trajectory():
    """Generate trajectory plots for both networks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Network 1: Z = 4 * log2(16) = 16 ---
    crn1 = make_crn_x_times_log2y(4, 16)
    rng1 = np.random.default_rng(42)
    final1, traj1 = crn1.simulate(max_steps=1000000, max_time=50000, rng=rng1,
                                   track=['x', 'y', 'z', 'w'])
    print(f"Network 1 (Z=4*log2(16)=16): final z={final1['z']}, w={final1['w']}, steps={len(traj1[0])-1}")

    ax = axes[0]
    s_vals = np.array(traj1[1])
    colors = {'x': 'blue', 'y': 'green', 'z': 'red', 'w': 'orange'}
    labels = {'x': 'X (input)', 'y': 'Y (input)', 'z': 'Z (output)', 'w': 'W (intermediate)'}
    for i, name in enumerate(traj1[2]):
        ax.plot(traj1[0], s_vals[:, i], color=colors[name], label=labels[name],
                linewidth=1.2, alpha=0.8)
    ax.set_xlabel('Simulation Time')
    ax.set_ylabel('Molecule Count')
    ax.set_title(r'$Z = X_0 \cdot \log_2(Y_0)$' + f'\nX₀=4, Y₀=16 → Z={final1["z"]} (expected 16)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Network 2: Y = 2^(log2(16)) = 16 ---
    crn2 = make_crn_2_to_log2x(16)
    rng2 = np.random.default_rng(42)
    final2, traj2 = crn2.simulate(max_steps=1000000, max_time=50000, rng=rng2,
                                   track=['x', 'y', 'w'])
    print(f"Network 2 (Y=2^(log2(16))=16): final y={final2['y']}, w={final2['w']}, steps={len(traj2[0])-1}")

    ax = axes[1]
    s_vals = np.array(traj2[1])
    colors2 = {'x': 'blue', 'y': 'red', 'w': 'orange'}
    labels2 = {'x': 'X (input)', 'y': 'Y (output)', 'w': 'W (intermediate)'}
    for i, name in enumerate(traj2[2]):
        ax.plot(traj2[0], s_vals[:, i], color=colors2[name], label=labels2[name],
                linewidth=1.2, alpha=0.8)
    ax.set_xlabel('Simulation Time')
    ax.set_ylabel('Molecule Count')
    ax.set_title(r'$Y = 2^{\log_2(X_0)}$' + f'\nX₀=16 → Y={final2["y"]} (expected 16)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/crn_trajectories.png', dpi=150, bbox_inches='tight')
    print("Plot saved to crn_trajectories.png")
    plt.close()


def write_aleae_files():
    """Write .in and .r files compatible with the aleae simulator."""
    # Network 1: Z = X * log2(Y), test: X=4, Y=16
    with open('/home/claude/crn1.in', 'w') as f:
        f.write("x 4 N\ny 16 N\nz 0 N\nb 1 N\na 0 N\nc 0 N\nyp 0 N\nw 0 N\nd 0 N\nxp 0 N\nzp 0 N\n")
    with open('/home/claude/crn1.r', 'w') as f:
        f.write(f"b 1 : a 1 b 1 : {LOG_SLOW}\n")
        f.write(f"a 1 y 2 : c 1 yp 1 a 1 : {R_FASTER}\n")
        f.write(f"c 2 : c 1 : {R_FASTER}\n")
        f.write(f"a 1 : : {R_FAST}\n")
        f.write(f"yp 1 : y 1 : {LOG_MEDIUM}\n")
        f.write(f"c 1 : w 1 : {LOG_MEDIUM}\n")
        f.write(f"w 1 : d 1 : {MOD2_SLOW}\n")
        f.write(f"d 1 x 1 : d 1 xp 1 zp 1 : {R_FASTER}\n")
        f.write(f"d 1 : : {R_FAST}\n")
        f.write(f"xp 1 : x 1 : {MOD2_MEDIUM}\n")
        f.write(f"zp 1 : z 1 : {MOD2_MEDIUM}\n")

    # Network 2: Y = 2^(log2(X)), test: X=16
    with open('/home/claude/crn2.in', 'w') as f:
        f.write("x 16 N\ny 1 N\nb 1 N\na 0 N\nc 0 N\nxp 0 N\nw 0 N\ne 0 N\nyp 0 N\n")
    with open('/home/claude/crn2.r', 'w') as f:
        f.write(f"b 1 : a 1 b 1 : {LOG_SLOW}\n")
        f.write(f"a 1 x 2 : c 1 xp 1 a 1 : {R_FASTER}\n")
        f.write(f"c 2 : c 1 : {R_FASTER}\n")
        f.write(f"a 1 : : {R_FAST}\n")
        f.write(f"xp 1 : x 1 : {LOG_MEDIUM}\n")
        f.write(f"c 1 : w 1 : {LOG_MEDIUM}\n")
        f.write(f"w 1 : e 1 : {MOD2_SLOW}\n")
        f.write(f"e 1 y 1 : e 1 yp 2 : {R_FASTER}\n")
        f.write(f"e 1 : : {R_FAST}\n")
        f.write(f"yp 1 : y 1 : {MOD2_MEDIUM}\n")
    print("Aleae files written: crn1.in, crn1.r, crn2.in, crn2.r\n")


if __name__ == "__main__":
    t0 = time.time()
    write_aleae_files()
    test_network1(num_trials=30)
    test_network2(num_trials=30)
    demo_trajectory()
    print(f"\nTotal time: {time.time()-t0:.1f}s")
