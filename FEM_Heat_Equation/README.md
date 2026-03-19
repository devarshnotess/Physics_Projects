
# FEM Heat Equation Solver

## Overview
This project implements a two-dimensional steady-state heat equation solver using the Finite Element Method (FEM). The formulation is derived from first principles using the weak (variational) form of the governing differential equation.

## Problem Statement
The steady-state heat equation is given by:

∇²T = 0

The goal is to compute the temperature distribution over a 2D domain subject to appropriate boundary conditions.

## Methodology
- Derived the weak formulation of the governing equation
- Discretized the domain using linear triangular elements
- Constructed element-level stiffness matrices
- Assembled the global system of equations
- Applied Dirichlet boundary conditions
- Solved the resulting linear system

## Features
- Finite Element formulation based on physical conservation laws
- Support for triangular mesh elements
- Boundary condition implementation
- Gradient-based and energy-based analysis
- Mesh sensitivity and convergence study

## Results
- Temperature distribution across the domain
- Gradient analysis to study heat flow
- Convergence behavior with mesh refinement

All detailed results, derivations, and analysis are included in the report.

## Tech Stack
- Python
- NumPy
- SciPy
- Matplotlib

## Files
- `FEM_solver.py` — FEM implementation
- `FEM_SOLVER.pdf` — detailed derivation and analysis

## Notes
This project focuses on combining numerical methods with physical interpretation, emphasizing stability and convergence of the solution.

