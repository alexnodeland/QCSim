"""Grover's algorithm for a 2-qubit search (using 3 qubits total: 2 search + 1 ancilla)."""

import qcsim

# Initialize 3-qubit register: 2 search qubits + 1 ancilla
STATE = '001'
q = qcsim.QuantumRegister(STATE)
c = qcsim.ClassicalRegister(3)
qc = qcsim.QuantumCircuit(q, c)

# Superposition
qc.H(0)
qc.H(1)
qc.H(2)

# Oracle: mark target state
qc.X(0)
qc.CCX(0, 1, 2)
qc.X(0)

# Diffusion operator
qc.H(0)
qc.H(1)
qc.X(0)
qc.X(1)
qc.CZ(0, 1)
qc.X(0)
qc.X(1)
qc.H(0)
qc.H(1)

# Results
print("State vector:")
print(qcsim.Result.get_statevector(qc))
print("\nMeasurement counts (1024 shots):")
print(qc.measure(1024))
