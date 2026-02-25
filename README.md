# EE 5393 Homework #1 — Circuits, Computation, and Biology

University of Minnesota, Winter 2026

## Structure

```
hw1_problem1.py        # Problem 1: Gillespie simulator + exact enumeration
hw1_problem2.py        # Problem 2: Lambda phage (wrapper for aleae simulator)
hw1_problem3.py        # Problem 3: CRN synthesis (Z=X·log₂Y, Y=2^log₂X)
aleae_files/           # Aleae-format input files for Problem 3
  crn1.in, crn1.r      #   Part 1: Z = X·log₂Y
  crn2.in, crn2.r      #   Part 2: Y = 2^(log₂X)
figures/               # Generated plots
  lambda_plot.png       #   Problem 2 result
  crn_trajectories.png  #   Problem 3 trajectories
```

## How to Run

**Problem 1** (self-contained):
```bash
python3 hw1_problem1.py
```

**Problem 2** (requires compiled `aleae` binary):
```bash
python3 hw1_problem2.py 1000 /path/to/aleae/vanilla/
```

**Problem 3** (self-contained):
```bash
python3 hw1_problem3.py
```

## AI Disclosure

All code and the written report were generated with the assistance of Claude (AI). See the PDF report for detailed per-problem AI disclosure statements.
