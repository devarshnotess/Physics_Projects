# Quantum Noise Modeling using Qiskit

## Overview
This project explores the impact of realistic noise on quantum circuits using Qiskit. It focuses on modeling decoherence and gate errors and analyzing their effect on quantum computation outcomes.

## Objective
To simulate quantum circuits under different noise models and study how noise affects:
- measurement distributions
- fidelity of quantum states
- overall computation accuracy

## Methodology
- Constructed quantum circuits (e.g., Bell states)
- Implemented noise models using Qiskit Aer:
  - Depolarizing noise
  - Amplitude damping
- Simulated circuits using noisy simulators
- Compared results with ideal (noise-free) simulations

## Noise Models Studied

### Depolarizing Noise
- Models random errors in quantum gates
- Causes loss of quantum information

### Amplitude Damping
- Models energy dissipation
- Represents relaxation processes in qubits

## Analysis
- Compared ideal vs noisy output distributions
- Studied error growth with increasing noise strength
- Evaluated degradation in fidelity
- Observed impact of circuit depth on noise accumulation

## Results
- Statistical comparison of measurement outcomes
- Visualization of noise effects on quantum states
- Quantitative analysis of fidelity decay

All detailed simulations and results are included in the report.

## Tech Stack
- Python
- Qiskit
- NumPy
- Matplotlib

## Files
- `main.py` — simulation and noise modeling code
- `Qiskit_Project.pdf` — detailed analysis and results

## Notes
This project demonstrates how noise impacts quantum computation and provides insights into the challenges of real quantum hardware.
