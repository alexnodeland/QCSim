# QCSim

QCSim is a simple quantum circuit simulator implemented in Python. It provides basic quantum gate operations and allows for the creation and manipulation of quantum circuits. The simulator is designed for educational purposes, helping users to understand the fundamentals of quantum computing.

## Features

- **Quantum Gates**: Implements common quantum gates such as X, Y, Z, H, S, T, and rotation gates.
- **Quantum Circuits**: Allows for the creation and manipulation of quantum circuits with various gate operations.
- **Measurement**: Simulate quantum measurement with configurable shot counts.
- **Non-adjacent qubit support**: Two-qubit gates (CX, CY, CZ) work on any pair of qubits, not just adjacent ones.
- **Quantum Devices**: Provides predefined quantum device topologies like IBM Q 20 Austin, IBM Q 16 Reuschlikon, and others.
- **Error Handling**: Includes custom exceptions for handling errors in input and gate operations.

## Installation

```bash
git clone https://github.com/alexnodeland/qcsim.git
cd qcsim
pip install .
```

For development (includes pytest):

```bash
pip install -e ".[dev]"
```

## Usage

### Example: Grover's Algorithm (2-qubit search)

This [example](grover_2_qubit.py) demonstrates how to use QCSim to implement Grover's algorithm for a 2-qubit search space (using 3 qubits total: 2 search qubits + 1 ancilla).

```python
import qcsim

q = qcsim.QuantumRegister('00')
qc = qcsim.QuantumCircuit(q)

qc.H(0)
qc.CX(0, 1)

print(qcsim.Result.get_statevector(qc))
print(qc.measure(1024))
```

## Quantum Gates

The following quantum gates are available in the simulator:

- **Single-qubit gates**: X, Y, Z, H, S, T
- **Rotation gates**: RX(theta), RY(theta), RZ(phi), RPHI(phi)
- **Two-qubit gates**: SWAP, CX, CY, CZ
- **Three-qubit gates**: CCX, CSWAP

## Quantum Devices

Predefined quantum device topologies available:

- IBM Q 20 Austin
- IBM Q 16 Reuschlikon
- IBM Q 5 Tenerife
- IBM Q 5 Yorktown
- Rigetti 19Q

## Testing

```bash
pytest
```

## Error Handling

Custom exceptions are provided to handle errors:

- `QCSimError`: Raised for errors in the input expressions or gate operations.
