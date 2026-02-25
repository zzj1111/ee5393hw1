"""
EE 5393 - Homework 1, Problem 2: Synthesizing Randomness (Lambda Phage)
======================================================================

This script runs the aleae stochastic simulator for the lambda phage model
with MOI = 1..10, and plots the probability of stealth vs hijack mode.

Requirements:
  - Compiled 'aleae' binary in the same directory as lambda.r and lambda.in
  - matplotlib (for plotting)

Usage:
  python3 hw1_problem2.py [num_trials] [aleae_dir]

  Default: 1000 trials, aleae in ./aleae/vanilla/

AI Disclosure: This script was generated with the assistance of Claude (AI).
"""

import subprocess
import sys
import os
import re
import tempfile

def run_lambda_simulation(aleae_dir, moi, num_trials):
    """Run aleae for a given MOI value and return (stealth_count, hijack_count)."""
    
    # Read template .in file and modify MOI
    in_path = os.path.join(aleae_dir, "lambda.in")
    with open(in_path, "r") as f:
        content = f.read()
    
    # Replace MOI line
    content = re.sub(r'^MOI \d+ ', f'MOI {moi} ', content, flags=re.MULTILINE)
    
    # Write to temp file
    tmp_in = tempfile.NamedTemporaryFile(mode='w', suffix='.in', delete=False, dir=aleae_dir)
    tmp_in.write(content)
    tmp_in.close()
    
    try:
        aleae_bin = os.path.join(aleae_dir, "aleae")
        r_file = os.path.join(aleae_dir, "lambda.r")
        
        result = subprocess.run(
            [aleae_bin, tmp_in.name, r_file, str(num_trials), "-1", "0"],
            capture_output=True, text=True, timeout=600
        )
        output = result.stdout
        
        # Parse: "cI2 >= 145: NNN (XX.XXXX%)"
        stealth_match = re.search(r'cI2 >= 145: (\d+)', output)
        hijack_match = re.search(r'Cro2 >= 55: (\d+)', output)
        
        stealth = int(stealth_match.group(1)) if stealth_match else 0
        hijack = int(hijack_match.group(1)) if hijack_match else 0
        
        return stealth, hijack
    finally:
        os.unlink(tmp_in.name)


def main():
    num_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    aleae_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), "aleae", "vanilla")
    
    if not os.path.isfile(os.path.join(aleae_dir, "aleae")):
        print(f"Error: aleae binary not found in {aleae_dir}")
        print("Please compile it first: cd aleae/vanilla && make")
        sys.exit(1)
    
    print("=" * 65)
    print("Problem 2: Lambda Phage — Stealth vs Hijack Mode")
    print(f"  Trials per MOI: {num_trials}")
    print(f"  Stealth: cI2 >= 145  |  Hijack: Cro2 >= 55")
    print("=" * 65)
    print()
    
    moi_values = list(range(1, 11))
    stealth_probs = []
    hijack_probs = []
    
    print(f"{'MOI':>3} | {'Stealth':>10} | {'Hijack':>10} | {'Stealth%':>10} | {'Hijack%':>10}")
    print("-" * 55)
    
    for moi in moi_values:
        stealth, hijack = run_lambda_simulation(aleae_dir, moi, num_trials)
        s_pct = stealth / num_trials * 100
        h_pct = hijack / num_trials * 100
        stealth_probs.append(s_pct)
        hijack_probs.append(h_pct)
        
        print(f"{moi:>3} | {stealth:>10} | {hijack:>10} | {s_pct:>9.2f}% | {h_pct:>9.2f}%")
    
    print()
    
    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(moi_values, stealth_probs, 'b-o', label='Stealth (cI2 ≥ 145)', linewidth=2, markersize=8)
        ax.plot(moi_values, hijack_probs, 'r-s', label='Hijack (Cro2 ≥ 55)', linewidth=2, markersize=8)
        ax.set_xlabel('MOI (Multiplicity of Infection)', fontsize=12)
        ax.set_ylabel('Probability (%)', fontsize=12)
        ax.set_title('Lambda Phage: Stealth vs Hijack Mode', fontsize=14)
        ax.set_xticks(moi_values)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        plot_path = os.path.join(os.path.dirname(__file__), "lambda_plot.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {plot_path}")
        plt.close()
    except ImportError:
        print("matplotlib not available — skipping plot.")


if __name__ == "__main__":
    main()
