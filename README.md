# QCSim

QCSim is a simple quantum circuit simulator implemented in Python. It provides basic quantum gate operations and allows for the creation and manipulation of quantum circuits. The simulator is designed for educational purposes, helping users to understand the fundamentals of quantum computing.

## Features

- **Quantum Gates**: Implements common quantum gates such as X, Y, Z, H, S, T, and rotation gates.
- **Quantum Circuits**: Allows for the creation and manipulation of quantum circuits with various gate operations.
- **Quantum Devices**: Provides predefined quantum devices like IBM Q 20 Austin, IBM Q 16 Reuschlikon, and others.
- **Error Handling**: Includes custom exceptions for handling errors in input and gate operations.

## Installation

To use QCSim, simply clone the repository and ensure you have NumPy installed.

```bash
git clone <repository-url>
cd qcsim
pip install numpy
```

## Usage

### Example: Grover's Algorithm (2-qubit)

This [example](grover_2_qubit.py) demonstrates how to use QCSim to implement Grover's algorithm for a 2-qubit system.

## Quantum Gates

The following quantum gates are available in the simulator:

- **Single-qubit gates**: X, Y, Z, H, S, T
- **Rotation gates**: RX(theta), RY(theta), RZ(phi), RPHI(phi)
- **Two-qubit gates**: SWAP, CX, CY, CZ
- **Three-qubit gates**: CCX, CSWAP

## Quantum Devices

Predefined quantum devices available:

- IBM Q 20 Austin
- IBM Q 16 Reuschlikon
- IBM Q 5 Tenerife
- IBM Q 5 Yorktown
- Rigetti 19Q

## Error Handling

Custom exceptions are provided to handle errors:

- `QCSimError`: Raised for errors in the input expressions or gate operations.
