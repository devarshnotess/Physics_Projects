import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, amplitude_damping_error, phase_damping_error
from qiskit.quantum_info import Statevector, state_fidelity

# -----------------------------
# 1. CIRCUIT DEFINITIONS
# -----------------------------
def single_qubit_circuit():
    qc = QuantumCircuit(1)
    qc.h(0)
    return qc

def two_qubit_entangled():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    return qc

# -----------------------------
# 2. IDEAL SIMULATION
# -----------------------------
def get_ideal_state(qc):
    return Statevector.from_instruction(qc)

from math import gamma
# -----------------------------
# 3. NOISE MODELS
# -----------------------------
def create_noise_model(p_dep=0.01, gamma_amp=0.01, gamma_phase=0.01):
    noise_model = NoiseModel()

    # Combine errors properly instead of stacking blindly
    error = depolarizing_error(p_dep,1)
    error = error.compose(amplitude_damping_error(gamma_amp))
    error = error.compose(phase_damping_error(gamma_amp))

    noise_model.add_all_qubit_quantum_error(error, ['h', 'x'])
    return noise_model


# -----------------------------
# 4. RUN NOISY SIMULATION
# -----------------------------
def run_noisy_simulation(qc, noise_model, shots=1024):
    qc_meas = qc.copy()
    qc_meas.measure_all()

    sim = AerSimulator(noise_model=noise_model)
    tqc = transpile(qc_meas, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts()

    return counts



# -----------------------------
# 5. COUNTS TO PROBABILITY VECTOR
# -----------------------------
def counts_to_prob(counts, num_qubits):
    total = sum(counts.values())
    probs = np.zeros(2**num_qubits)

    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        probs[idx] = count / total

    return probs


# -----------------------------
# 6. FIDELITY FROM PROBABILITY
# -----------------------------
def classical_fidelity(p, q):
    return np.sum(np.sqrt(p * q))**2

# -----------------------------
# 7. PARAMETER SWEEP
# -----------------------------
noise_levels = np.linspace(0.0, 0.2, 10)
fidelities_single = []
fidelities_two = []

qc1 = single_qubit_circuit()
qc2 = two_qubit_entangled()

ideal1 = get_ideal_state(qc1)
ideal2 = get_ideal_state(qc2)

ideal_probs1 = np.abs(ideal1.data)**2
ideal_probs2 = np.abs(ideal2.data)**2

for p in noise_levels:
    noise_model = create_noise_model(p_dep=p, gamma_amp=p, gamma_phase=p)

    counts1 = run_noisy_simulation(qc1, noise_model)
    counts2 = run_noisy_simulation(qc2, noise_model)

    probs1 = counts_to_prob(counts1, 1)
    probs2 = counts_to_prob(counts2, 2)

    fid1 = classical_fidelity(ideal_probs1, probs1)
    fid2 = classical_fidelity(ideal_probs2, probs2)

    fidelities_single.append(fid1)
    fidelities_two.append(fid2)



# -----------------------------
# 8. PLOT: FIDELITY VS NOISE
# -----------------------------
plt.figure()
plt.plot(noise_levels, fidelities_single, label="Single Qubit")
plt.plot(noise_levels, fidelities_two, label="Two Qubit Entangled")
plt.xlabel("Noise Strength")
plt.ylabel("Fidelity")
plt.title("Fidelity vs Noise Strength")
plt.legend()
plt.grid()
plt.show()

# -----------------------------
# 9. IDEAL vs NOISY DISTRIBUTION
# -----------------------------
noise_model = create_noise_model(0.1, 0.1, 0.1)
counts = run_noisy_simulation(qc2, noise_model)
noisy_probs = counts_to_prob(counts, 2)

x = np.arange(len(ideal_probs2))

plt.figure()
plt.bar(x - 0.2, ideal_probs2, width=0.4, label="Ideal")
plt.bar(x + 0.2, noisy_probs, width=0.4, label="Noisy")
plt.xlabel("State Index")
plt.ylabel("Probability")
plt.title("Ideal vs Noisy Distribution (2-Qubit)")
plt.legend()
plt.show()


# -----------------------------
# 10. BASIC MEASUREMENT ERROR MITIGATION
# -----------------------------
def mitigate_counts(counts):
    total = sum(counts.values())
    mitigated = {k: v/total for k, v in counts.items()}
    return mitigated

mitigated_counts = mitigate_counts(counts)
mitigated_probs = counts_to_prob(mitigated_counts, 2)

plt.figure()
plt.bar(x - 0.25, ideal_probs2, width=0.25, label="Ideal")
plt.bar(x, noisy_probs, width=0.25, label="Noisy")
plt.bar(x + 0.25, mitigated_probs, width=0.25, label="Mitigated")
plt.xlabel("State Index")
plt.ylabel("Probability")
plt.title("Error Mitigation Effect")
plt.legend()
plt.show()
