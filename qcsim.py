import numpy as np


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class QCSimError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class Gate:
    """Quantum gate definitions.

    Fixed gates are cached as class-level constants.
    Parameterized gates are available as static methods.
    """

    # Single-qubit gates
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=complex)
    S = np.array([[1, 0], [0, 1j]], dtype=complex)
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)

    # Two-qubit gates
    SWAP = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex)
    CX = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]], dtype=complex)
    CY = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 0, -1j],
                   [0, 0, 1j, 0]], dtype=complex)
    CZ = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, -1]], dtype=complex)

    # Three-qubit gates
    CCX = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0]], dtype=complex)
    CSWAP = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1]], dtype=complex)

    @staticmethod
    def RX(theta):
        c = np.cos(theta / 2)
        s = -1j * np.sin(theta / 2)
        return np.array([[c, s], [s, c]], dtype=complex)

    @staticmethod
    def RY(theta):
        c = np.cos(theta / 2)
        s = np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    @staticmethod
    def RZ(phi):
        em = np.exp(-1j * phi / 2)
        ep = np.exp(1j * phi / 2)
        return np.array([[em, 0], [0, ep]], dtype=complex)

    @staticmethod
    def RPHI(phi):
        ep = np.exp(1j * phi)
        return np.array([[1, 0], [0, ep]], dtype=complex)


class Device:
    """Predefined quantum hardware topology adjacency matrices."""

    IBM_Q_20_AUSTIN = np.array([
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]])

    IBM_Q_16_REUSCHLIKON = np.array([
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])

    IBM_Q_5_TENERIFE = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0]])

    IBM_Q_5_YORKTOWN = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 1, 0, 1],
        [0, 0, 1, 1, 0]])

    RIGETTI_19Q = np.array([
        [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]])


class QuantumCircuit:
    def __init__(self, q, c=None):
        self._size = len(q.get_qreg)
        self._state_vec = self._make_statevector(q.get_qreg)

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, val):
        self._size = val

    @property
    def state_vec(self):
        return self._state_vec

    @state_vec.setter
    def state_vec(self, val):
        self._state_vec = val

    def _make_statevector(self, sb):
        if sb[0] == 0:
            sv = np.array([[1], [0]], dtype=complex)
        else:
            sv = np.array([[0], [1]], dtype=complex)
        for i in range(1, self.size):
            if sb[i] == 0:
                sv = np.kron(sv, np.array([[1], [0]], dtype=complex))
            else:
                sv = np.kron(sv, np.array([[0], [1]], dtype=complex))
        return sv

    def _embed(self, gate, index, sz):
        if index == 0:
            m = np.kron(gate, np.identity(int(np.power(2, self.size - sz))))
        elif index == self.size - sz:
            m = np.kron(np.identity(int(np.power(2, self.size - sz))), gate)
        else:
            m = np.kron(np.identity(int(np.power(2, index))), gate)
            m = np.kron(m, np.identity(int(np.power(2, self.size - index - sz))))
        return m

    def _swap_adjacent(self, i):
        """Swap adjacent qubits i and i+1."""
        swap = self._embed(Gate.SWAP, i, 2)
        self.state_vec = swap.dot(self.state_vec)

    def _route_swap(self, ctl, tgt):
        """Use a chain of SWAP gates to bring non-adjacent qubits together,
        apply the operation, then swap them back."""
        swaps = []
        if tgt < ctl:
            # Move target up to ctl-1
            for i in range(tgt, ctl - 1):
                self._swap_adjacent(i)
                swaps.append(i)
            return ctl - 1, ctl, swaps
        else:
            # Move target down to ctl+1
            for i in range(tgt - 1, ctl, -1):
                self._swap_adjacent(i)
                swaps.append(i)
            return ctl, ctl + 1, swaps

    def _unswap(self, swaps):
        """Reverse the SWAP chain to restore qubit ordering."""
        for i in reversed(swaps):
            self._swap_adjacent(i)

    def RX(self, theta, q):
        rx = self._embed(Gate.RX(theta), q, 1)
        self.state_vec = rx.dot(self.state_vec)

    def RY(self, theta, q):
        ry = self._embed(Gate.RY(theta), q, 1)
        self.state_vec = ry.dot(self.state_vec)

    def RZ(self, phi, q):
        rz = self._embed(Gate.RZ(phi), q, 1)
        self.state_vec = rz.dot(self.state_vec)

    def RPHI(self, phi, q):
        rphi = self._embed(Gate.RPHI(phi), q, 1)
        self.state_vec = rphi.dot(self.state_vec)

    def X(self, q):
        x = self._embed(Gate.X, q, 1)
        self.state_vec = x.dot(self.state_vec)

    def Y(self, q):
        y = self._embed(Gate.Y, q, 1)
        self.state_vec = y.dot(self.state_vec)

    def Z(self, q):
        z = self._embed(Gate.Z, q, 1)
        self.state_vec = z.dot(self.state_vec)

    def H(self, q):
        h = self._embed(Gate.H, q, 1)
        self.state_vec = h.dot(self.state_vec)

    def S(self, q):
        s = self._embed(Gate.S, q, 1)
        self.state_vec = s.dot(self.state_vec)

    def T(self, q):
        t = self._embed(Gate.T, q, 1)
        self.state_vec = t.dot(self.state_vec)

    def SWAP(self, ctl, tgt):
        if abs(ctl - tgt) == 1:
            swap = self._embed(Gate.SWAP, min(ctl, tgt), 2)
            self.state_vec = swap.dot(self.state_vec)
        else:
            # Route through chain of adjacent SWAPs
            _, _, swaps = self._route_swap(ctl, tgt)
            self._swap_adjacent(min(ctl, tgt) if abs(ctl - tgt) == 1 else swaps[-1] if swaps else min(ctl, tgt))
            self._unswap(swaps)

    def CX(self, ctl, tgt):
        if abs(ctl - tgt) == 1:
            if tgt < ctl:
                self._swap_adjacent(tgt)
            cx = self._embed(Gate.CX, min(ctl, tgt), 2)
            self.state_vec = cx.dot(self.state_vec)
            if tgt < ctl:
                self._swap_adjacent(tgt)
        else:
            new_ctl, new_tgt, swaps = self._route_swap(ctl, tgt)
            if new_tgt < new_ctl:
                self._swap_adjacent(new_tgt)
            cx = self._embed(Gate.CX, min(new_ctl, new_tgt), 2)
            self.state_vec = cx.dot(self.state_vec)
            if new_tgt < new_ctl:
                self._swap_adjacent(new_tgt)
            self._unswap(swaps)

    def CY(self, ctl, tgt):
        if abs(ctl - tgt) == 1:
            if tgt < ctl:
                self._swap_adjacent(tgt)
            cy = self._embed(Gate.CY, min(ctl, tgt), 2)
            self.state_vec = cy.dot(self.state_vec)
            if tgt < ctl:
                self._swap_adjacent(tgt)
        else:
            new_ctl, new_tgt, swaps = self._route_swap(ctl, tgt)
            if new_tgt < new_ctl:
                self._swap_adjacent(new_tgt)
            cy = self._embed(Gate.CY, min(new_ctl, new_tgt), 2)
            self.state_vec = cy.dot(self.state_vec)
            if new_tgt < new_ctl:
                self._swap_adjacent(new_tgt)
            self._unswap(swaps)

    def CZ(self, ctl, tgt):
        if abs(ctl - tgt) == 1:
            if tgt < ctl:
                self._swap_adjacent(tgt)
            cz = self._embed(Gate.CZ, min(ctl, tgt), 2)
            self.state_vec = cz.dot(self.state_vec)
            if tgt < ctl:
                self._swap_adjacent(tgt)
        else:
            new_ctl, new_tgt, swaps = self._route_swap(ctl, tgt)
            if new_tgt < new_ctl:
                self._swap_adjacent(new_tgt)
            cz = self._embed(Gate.CZ, min(new_ctl, new_tgt), 2)
            self.state_vec = cz.dot(self.state_vec)
            if new_tgt < new_ctl:
                self._swap_adjacent(new_tgt)
            self._unswap(swaps)

    def CCX(self, ctl1, ctl2, tgt):
        if abs(ctl1 - ctl2) != 1 or tgt - max(ctl1, ctl2) != 1:
            raise QCSimError('CCX', 'Controls and target should be consecutive qubits!')
        ccx = self._embed(Gate.CCX, min(ctl1, ctl2), 3)
        self.state_vec = ccx.dot(self.state_vec)

    def CSWAP(self, ctl, tgt1, tgt2):
        if abs(tgt1 - tgt2) != 1 or min(tgt1, tgt2) - ctl != 1:
            raise QCSimError('CSWAP', 'Control and targets should be consecutive qubits!')
        cswap = self._embed(Gate.CSWAP, ctl, 3)
        self.state_vec = cswap.dot(self.state_vec)

    def measure(self, shots=1024):
        """Simulate measurement of all qubits.

        Returns a dictionary mapping bitstring outcomes to their counts.
        """
        probs = np.abs(self.state_vec.flatten()) ** 2
        probs = np.real(probs)
        probs = probs / probs.sum()  # Normalize to handle floating point drift
        num_states = len(probs)
        num_qubits = self.size
        indices = np.random.choice(num_states, size=shots, p=probs)
        counts = {}
        for idx in indices:
            bitstring = format(idx, f'0{num_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts


class QuantumRegister:
    def __init__(self, qr):
        if isinstance(qr, int):
            self._q = np.zeros(qr)[np.newaxis].T
        elif isinstance(qr, str):
            if set(qr) <= set('01'):
                self._q = np.array(list(map(int, list(qr))))[np.newaxis].T
            else:
                raise QCSimError('QuantumRegister', 'If specifying explicit state(str), please input string of 1s and 0s')
        else:
            raise QCSimError('QuantumRegister', 'Please input size(int) or explicit state(str)')

    @property
    def get_qreg(self):
        return self._q


class ClassicalRegister:
    def __init__(self, cr):
        if isinstance(cr, int):
            self._c = np.zeros(cr)[np.newaxis].T
        elif isinstance(cr, str):
            if set(cr) <= set('01'):
                self._c = np.array(list(map(int, list(cr))))[np.newaxis].T
            else:
                raise QCSimError('ClassicalRegister', 'If specifying explicit state(str), please input string of 1s and 0s')
        else:
            raise QCSimError('ClassicalRegister', 'Please input size(int) or explicit state(str)')

    @property
    def get_creg(self):
        return self._c


class Result:
    @staticmethod
    def get_statevector(circuit):
        return circuit.state_vec

    @staticmethod
    def get_PDM(circuit):
        sv = Result.get_statevector(circuit)
        return sv.dot(sv.conj().T)

    @staticmethod
    def get_ZPV(circuit):
        return np.diagonal(Result.get_PDM(circuit))[np.newaxis].T
