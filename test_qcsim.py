"""Tests for QCSim quantum circuit simulator."""

import numpy as np
import pytest
import qcsim
from qcsim import Gate, QuantumRegister, ClassicalRegister, QuantumCircuit, Result, QCSimError


# --- Gate matrix tests ---

class TestGateMatrices:
    def test_x_gate_is_unitary(self):
        assert np.allclose(Gate.X @ Gate.X, np.eye(2))

    def test_y_gate_is_unitary(self):
        assert np.allclose(Gate.Y @ Gate.Y, np.eye(2))

    def test_z_gate_is_unitary(self):
        assert np.allclose(Gate.Z @ Gate.Z, np.eye(2))

    def test_h_gate_is_unitary(self):
        assert np.allclose(Gate.H @ Gate.H, np.eye(2))

    def test_s_gate_squared_is_z(self):
        assert np.allclose(Gate.S @ Gate.S, Gate.Z)

    def test_t_gate_squared_is_s(self):
        assert np.allclose(Gate.T @ Gate.T, Gate.S)

    def test_swap_is_involution(self):
        assert np.allclose(Gate.SWAP @ Gate.SWAP, np.eye(4))

    def test_cx_is_unitary(self):
        assert np.allclose(Gate.CX @ Gate.CX, np.eye(4))

    def test_ccx_is_unitary(self):
        assert np.allclose(Gate.CCX @ Gate.CCX, np.eye(8))

    def test_rx_at_pi_is_minus_i_x(self):
        rx_pi = Gate.RX(np.pi)
        expected = -1j * Gate.X
        assert np.allclose(rx_pi, expected)

    def test_ry_at_pi_is_minus_i_y(self):
        ry_pi = Gate.RY(np.pi)
        expected = np.array([[0, -1], [1, 0]], dtype=complex)
        assert np.allclose(ry_pi, expected)

    def test_rz_at_pi_is_proportional_to_z(self):
        rz_pi = Gate.RZ(np.pi)
        # RZ(pi) = diag(e^{-i*pi/2}, e^{i*pi/2}) = diag(-i, i)
        expected = np.array([[-1j, 0], [0, 1j]], dtype=complex)
        assert np.allclose(rz_pi, expected)

    def test_rphi_at_pi_gives_z(self):
        rphi_pi = Gate.RPHI(np.pi)
        expected = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.allclose(rphi_pi, expected)


# --- Register tests ---

class TestRegisters:
    def test_quantum_register_int(self):
        qr = QuantumRegister(3)
        assert qr.get_qreg.shape == (3, 1)
        assert np.allclose(qr.get_qreg, np.zeros((3, 1)))

    def test_quantum_register_str(self):
        qr = QuantumRegister('101')
        expected = np.array([[1], [0], [1]])
        assert np.allclose(qr.get_qreg, expected)

    def test_quantum_register_invalid_str(self):
        with pytest.raises(QCSimError):
            QuantumRegister('abc')

    def test_quantum_register_invalid_type(self):
        with pytest.raises(QCSimError):
            QuantumRegister(3.5)

    def test_classical_register_int(self):
        cr = ClassicalRegister(4)
        assert cr.get_creg.shape == (4, 1)

    def test_classical_register_str(self):
        cr = ClassicalRegister('110')
        expected = np.array([[1], [1], [0]])
        assert np.allclose(cr.get_creg, expected)

    def test_classical_register_invalid_str(self):
        with pytest.raises(QCSimError):
            ClassicalRegister('xyz')

    def test_classical_register_invalid_type(self):
        with pytest.raises(QCSimError):
            ClassicalRegister([1, 0])


# --- Circuit tests ---

class TestCircuit:
    def _make_circuit(self, state='0'):
        q = QuantumRegister(state)
        return QuantumCircuit(q)

    def test_initial_state_zero(self):
        qc = self._make_circuit('0')
        sv = Result.get_statevector(qc)
        assert np.allclose(sv, np.array([[1], [0]]))

    def test_initial_state_one(self):
        qc = self._make_circuit('1')
        sv = Result.get_statevector(qc)
        assert np.allclose(sv, np.array([[0], [1]]))

    def test_x_flips_zero_to_one(self):
        qc = self._make_circuit('0')
        qc.X(0)
        sv = Result.get_statevector(qc)
        assert np.allclose(sv, np.array([[0], [1]]))

    def test_x_twice_is_identity(self):
        qc = self._make_circuit('0')
        qc.X(0)
        qc.X(0)
        sv = Result.get_statevector(qc)
        assert np.allclose(sv, np.array([[1], [0]]))

    def test_h_twice_is_identity(self):
        qc = self._make_circuit('0')
        qc.H(0)
        qc.H(0)
        sv = Result.get_statevector(qc)
        assert np.allclose(sv, np.array([[1], [0]]))

    def test_h_creates_superposition(self):
        qc = self._make_circuit('0')
        qc.H(0)
        sv = Result.get_statevector(qc)
        expected = np.array([[1], [1]]) / np.sqrt(2)
        assert np.allclose(sv, expected)

    def test_cnot_entanglement(self):
        """H on q0 then CNOT creates Bell state (|00> + |11>)/sqrt(2)."""
        qc = self._make_circuit('00')
        qc.H(0)
        qc.CX(0, 1)
        sv = Result.get_statevector(qc)
        expected = np.array([[1], [0], [0], [1]]) / np.sqrt(2)
        assert np.allclose(sv, expected)

    def test_cx_non_adjacent(self):
        """CX between qubit 0 and qubit 2 (non-adjacent) on a 3-qubit circuit."""
        qc = self._make_circuit('100')
        qc.CX(0, 2)
        sv = Result.get_statevector(qc)
        # |100> with CX(0,2) should give |101>
        expected = np.zeros((8, 1))
        expected[5] = 1  # |101> = index 5
        assert np.allclose(sv, expected)

    def test_cz_non_adjacent(self):
        """CZ between non-adjacent qubits."""
        # CZ on |11> should give -|11>, test on 3-qubit with qubits 0 and 2
        qc = self._make_circuit('101')
        qc.CZ(0, 2)
        sv = Result.get_statevector(qc)
        expected = np.zeros((8, 1), dtype=complex)
        expected[5] = -1  # |101> gets phase -1
        assert np.allclose(sv, expected)

    def test_swap_adjacent(self):
        qc = self._make_circuit('10')
        qc.SWAP(0, 1)
        sv = Result.get_statevector(qc)
        expected = np.array([[0], [1], [0], [0]])  # |01>
        assert np.allclose(sv, expected)

    def test_z_gate(self):
        qc = self._make_circuit('1')
        qc.Z(0)
        sv = Result.get_statevector(qc)
        assert np.allclose(sv, np.array([[0], [-1]]))

    def test_s_gate(self):
        qc = self._make_circuit('1')
        qc.S(0)
        sv = Result.get_statevector(qc)
        assert np.allclose(sv, np.array([[0], [1j]]))

    def test_t_gate(self):
        qc = self._make_circuit('1')
        qc.T(0)
        sv = Result.get_statevector(qc)
        expected = np.array([[0], [np.exp(1j * np.pi / 4)]])
        assert np.allclose(sv, expected)

    def test_rx_gate(self):
        qc = self._make_circuit('0')
        qc.RX(np.pi, 0)
        sv = Result.get_statevector(qc)
        # RX(pi)|0> = -i|1>
        expected = np.array([[0], [-1j]])
        assert np.allclose(sv, expected)

    def test_ry_gate(self):
        qc = self._make_circuit('0')
        qc.RY(np.pi, 0)
        sv = Result.get_statevector(qc)
        # RY(pi)|0> = |1>
        expected = np.array([[0], [1]], dtype=complex)
        assert np.allclose(sv, expected, atol=1e-10)

    def test_rz_gate(self):
        qc = self._make_circuit('1')
        qc.RZ(np.pi, 0)
        sv = Result.get_statevector(qc)
        expected = np.array([[0], [1j]], dtype=complex)
        assert np.allclose(sv, expected)

    def test_rphi_gate(self):
        qc = self._make_circuit('1')
        qc.RPHI(np.pi, 0)
        sv = Result.get_statevector(qc)
        expected = np.array([[0], [-1]], dtype=complex)
        assert np.allclose(sv, expected, atol=1e-10)

    def test_ccx_gate(self):
        qc = self._make_circuit('110')
        qc.CCX(0, 1, 2)
        sv = Result.get_statevector(qc)
        expected = np.zeros((8, 1))
        expected[7] = 1  # |111>
        assert np.allclose(sv, expected)

    def test_ccx_no_flip(self):
        qc = self._make_circuit('100')
        qc.CCX(0, 1, 2)
        sv = Result.get_statevector(qc)
        expected = np.zeros((8, 1))
        expected[4] = 1  # stays |100>
        assert np.allclose(sv, expected)

    def test_classical_register_optional(self):
        """QuantumCircuit should work without a ClassicalRegister."""
        q = QuantumRegister('00')
        qc = QuantumCircuit(q)
        qc.H(0)
        sv = Result.get_statevector(qc)
        assert sv.shape == (4, 1)


# --- Measurement tests ---

class TestMeasurement:
    def test_measure_deterministic(self):
        """Measuring |0> should always give '0'."""
        q = QuantumRegister('0')
        qc = QuantumCircuit(q)
        counts = qc.measure(100)
        assert counts == {'0': 100}

    def test_measure_deterministic_two_qubit(self):
        """Measuring |10> should always give '10'."""
        q = QuantumRegister('10')
        qc = QuantumCircuit(q)
        counts = qc.measure(100)
        assert counts == {'10': 100}

    def test_measure_superposition(self):
        """H|0> should give roughly 50/50 split."""
        q = QuantumRegister('0')
        qc = QuantumCircuit(q)
        qc.H(0)
        counts = qc.measure(10000)
        assert '0' in counts and '1' in counts
        # Allow wide margin for randomness
        assert counts['0'] > 4000
        assert counts['1'] > 4000

    def test_measure_total_shots(self):
        q = QuantumRegister('0')
        qc = QuantumCircuit(q)
        qc.H(0)
        counts = qc.measure(500)
        assert sum(counts.values()) == 500


# --- Result tests ---

class TestResult:
    def test_pdm_pure_state(self):
        q = QuantumRegister('0')
        qc = QuantumCircuit(q)
        pdm = Result.get_PDM(qc)
        # |0><0| = [[1,0],[0,0]]
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        assert np.allclose(pdm, expected)

    def test_zpv_pure_state(self):
        q = QuantumRegister('0')
        qc = QuantumCircuit(q)
        zpv = Result.get_ZPV(qc)
        expected = np.array([[1], [0]])
        assert np.allclose(zpv, expected)

    def test_zpv_superposition(self):
        q = QuantumRegister('0')
        qc = QuantumCircuit(q)
        qc.H(0)
        zpv = Result.get_ZPV(qc)
        expected = np.array([[0.5], [0.5]])
        assert np.allclose(zpv, expected)
