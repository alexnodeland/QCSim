import numpy as np

class Error(Exception):
    """Base class for exceptions in this module."""
    pass

class EntropicaError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expression -- input expression in which the error occurred
        message -- explanation of the error
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message

def constant(f):
    def fset(self, value):
        raise TypeError
    def fget(self):
        return f()
    return property(fget, fset)

class Gate(object):
    @constant
    def X():
        _x = np.matrix([[0,1],[1,0]])
        return _x
    
    @constant
    def Y():
        _y = np.matrix([[0,-1j],[1j,0]])
        return _y
    
    @constant
    def Z():
        _z = np.matrix([[1,0],[0,-1]])
        return _z
    
    @constant
    def H():
        _h = 1/np.sqrt(2)*np.matrix([[1,1],[1,-1]])
        return _h
    
    @constant
    def S():
        _s = np.matrix([[1,0],[0,1j]])
        return _s
    
    @constant
    def T():
        _t = np.matrix([[1,0],[0,np.exp(1j*np.pi/4)]])
        return -t
    
    @staticmethod
    def RX(theta):
        _c = np.cos(theta/2)
        _s = -1j*np.sin(theta/2)
        _rx = np.matrix([[_c,_s],[_s,_c]])
        return _rx
    
    @staticmethod
    def RY(theta):
        _c = np.cos(theta/2)
        _s = np.sin(theta/2)
        _ry = np.matrix([[_c,-1*_s],[_s,_c]])
        return _ry
    
    @staticmethod
    def RZ(phi):
        _em = np.exp(-1j*phi/2)
        _ep = np.exp(1j*phi/2)
        _rz = np.matrix([[_em,0],[0,_ep]])
        return _rz
    
    @staticmethod
    def RPHI(phi):
        _ep = np.exp(1j*phi)
        _rphi = np.matrix([[1,0],[0,_ep]])
        return _rz
    
    @constant
    def SWAP():
        _swap = np.matrix([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
        return _swap
    
    @constant
    def CX():
        _cx = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        return _cx
    
    @constant
    def CY():
        _cy = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,0,-1j],[0,0,1j,0]])
        return _cy
    
    @constant
    def CZ(): 
        _cz = np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]])
        return _cz
    
    @constant
    def CCX():
        _ccx = np.matrix([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]])
        return _ccx
    
    @constant 
    def CSWAP():
        _cswap = np.matrix([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1]])
        return _cswap
    
class Device(object):
    @constant
    def IBM_Q_20_AUSTIN():
        _ibm_q_20_austin = np.matrix([[0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                      [1,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                      [0,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],                    
                                      [0,0,1,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                                      [0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                                      [1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0],
                                      [0,1,1,0,0,1,0,1,0,0,1,1,0,0,0,0,0,0,0,0],
                                      [0,1,1,0,0,0,1,0,1,0,0,0,1,1,0,0,0,0,0,0],
                                      [0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,0,0,0,0,0],
                                      [0,0,0,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0],
                                      [0,0,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,0,0,0],
                                      [0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,1,1,0,0],
                                      [0,0,0,0,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,0],
                                      [0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,1,1],
                                      [0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1],
                                      [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
                                      [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,0,0],
                                      [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,0],
                                      [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1],
                                      [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0]]) 
        return _ibm_q_20_austin

    @constant
    def IBM_Q_16_REUSCHLIKON():
        _ibm_q_16_reuschlikon = np.matrix([[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
                                           [1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                           [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1],
                                           [0,0,1,0,1,0,0,0,0,0,0,0,0,0,1,0],
                                           [0,0,0,1,0,1,0,0,0,0,0,0,0,1,0,0],
                                           [0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0],
                                           [0,0,0,0,0,1,0,1,0,0,0,1,0,0,0,0],
                                           [0,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0],
                                           [0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0],
                                           [0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0],
                                           [0,0,0,0,0,0,0,1,0,1,0,1,0,0,0,0],
                                           [0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0],
                                           [0,0,0,0,0,1,0,0,0,0,0,1,0,1,0,0],
                                           [0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0],
                                           [0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,1],
                                           [1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0]])
        return _ibm_q_16_reuschlikon
    
    @constant
    def IBM_Q_5_TENERIFE():
        _ibm_q_5_tenerife = np.matrix([[0,1,1,0,0],
                                       [1,0,1,0,0],
                                       [1,1,0,1,1],
                                       [0,0,1,0,1],
                                       [0,0,1,1,0]])
        return _ibm_q_5_tenerife

    @constant
    def IBM_Q_5_YORKTOWN():
        _ibm_q_5_yorktown = np.matrix([[0,1,1,0,0],
                                       [1,0,1,0,0],
                                       [1,1,0,1,1],
                                       [0,0,1,0,1],
                                       [0,0,1,1,0]])
        return _ibm_q_5_yorktown
    
    @constant
    def RIGETTI_19Q():
        _rigetti_19q = np.matrix([[0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0],                    
                                  [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                                  [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                  [1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0], 
                                  [0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
                                  [0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
                                  [0,0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
                                  [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
                                  [0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0,0],
                                  [0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,0],
                                  [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1],
                                  [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],
                                  [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0],
                                  [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0]])
        return _rigetti_19q 

class QuantumCircuit(object):
    def __init__(self, q, c):
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
    def state_vec(self,val):
        self._state_vec = val
            
    def _make_statevector(self, sb):
        if sb[0] == 0:
            _sv = np.array([[1],[0]])
        else:
            _sv = np.array([[0],[1]])
        for i in range(1, self.size):
            if sb[i] == 0:
                _sv = np.kron(_sv, np.array([[1],[0]]))
            else:
                _sv = np.kron(_sv, np.array([[0],[1]]))
        return _sv
        
    def _embed(self, gate, index, sz):
        if index == 0:
            m = np.kron(gate, np.identity(np.power(2, self.size - sz)))
        elif index == self.size - 1:
            m = np.kron(np.identity(np.power(2, self.size - sz)), gate)
        else:
            m = np.kron(np.identity(np.power(2, index)), gate)
            m = np.kron(m, np.identity(np.power(2, self.size - index - sz)))
        return m
    
    def RX(self, theta, q):
        rx = self._embed(Gate().RX(theta),q,1)
        self.state_vec = rx.dot(self.state_vec)
    
    def RY(self, theta, q):
        ry = self._embed(Gate().RY(theta),q,1)
        self.state_vec = ry.dot(self.state_vec)
    
    def RZ(self, phi, q):
        rz = self._embed(Gate().RZ(theta),q,1)
        self.state_vec = rz.dot(self.state_vec)
        
    def RPHI(self, phi, q):
        rphi = self._embed(Gate().RPHI(phi),q,1)
        self.state_vec = rphi.dot(self.state_vec)
    
    def X(self, q):
        rx = self._embed(Gate().X,q,1)
        self.state_vec = rx.dot(self.state_vec)
    
    def Y(self, q):    
        ry = self._embed(Gate().Y,q,1)
        self.state_vec = ry.dot(self.state_vec)
       
    def Z(self, q):    
        rz = self._embed(Gate().Z,q,1)
        self.state_vec = rz.dot(self.state_vec)
    
    def H(self, q):
        h = self._embed(Gate().H,q,1)
        self.state_vec = h.dot(self.state_vec)
        
    def S(self, q):
        s= self._embed(Gate().S,q,1)
        self.state_vec = s.dot(self.state_vec)
        
    def T(self,q):
        t = self._embed(Gate().T,q,1)
        self.state_vec = t.dot(self.state_vec)
    
    def SWAP(self, ctl, tgt):
        if abs(ctl-tgt) != 1:
            raise EntropicaError('SWAP','Control and target should be consecutive qubits!')            
            quit()
        swap = self._embed(Gate().SWAP, min(ctl, tgt), 2)
        self.state_vec = swap.dot(self.state_vec)
        
    def CX(self, ctl, tgt):
        if abs(ctl-tgt) != 1:
            raise EntropicaError('CX','Control and target should be consecutive qubits!')
            quit()
        if tgt < ctl:
            self.SWAP(ctl, tgt)
        cx = self._embed(Gate().CX, min(ctl, tgt), 2)
        self.state_vec = cx.dot(self.state_vec)
        if tgt < ctl:
            self.SWAP(ctl, tgt)
        
    def CY(self, ctl, tgt):
        if abs(ctl-tgt) != 1:
            raise EntropicaError('CY','Control and target should be consecutive qubits!')
            quit()
        if tgt < ctl:
            self.SWAP(ctl, tgt)
        cy = self._embed(Gate().CY, min(ctl, tgt), 2)
        self.state_vec = cy.dot(self.state_vec)
        if tgt < ctl:
            self.SWAP(ctl, tgt)
        
    def CZ(self, ctl, tgt):
        if abs(ctl-tgt) != 1:
            raise EntropicaError('CZ','Control and target should be consecutive qubits!')
            quit()
        if tgt < ctl:
            self.SWAP(ctl, tgt)
        cz = self._embed(Gate().CZ, min(ctl, tgt), 2)
        self.state_vec = cz.dot(self.state_vec)
        if tgt < ctl:
            self.SWAP(ctl, tgt)
        
    def CCX(self, ctl1, ctl2, tgt):
        if abs(ctl1-ctl2) != 1 and tgt - max(ctl1,ctl2) != 1:
            raise EntropicaError('CCX','Controls and target should be consecutive qubits!')
            quit()
        ccx = self._embed(Gate().CCX, min(ctl1, ctl2), 3)
        self.state_vec = ccx.dot(self.state_vec)

    def CSWAP(self, ctl, tgt1, tgt2):
        if abs(tgt1-tgt2) != 1 and min(tgt1,tgt2) - ctl != 1:
            raise EntropicaError('CSWAP','Controls and target should be consecutive qubits!')
            quit()
        cswap = self._embed(Gate().CSWAP, ctl, 3)
        self.state_vec = cswap.dot(self.state_vec)    
 
class QuantumRegister(object):
    def __init__(self, qr):
        if isinstance(qr, int):
            self._q = np.zeros(qr)[np.newaxis].T
        elif isinstance(qr, str):
            if set(qr) <= set('01'):
                self._q = np.array(list(map(int,np.array(list(qr)))))[np.newaxis].T
            else:
                raise EntropicaError('QuantumRegister', 'If specifying explicit state(str), please input sting of 1"s and 0"s')
        else:
            raise EntropicaError('QuantumRegister','Please input size(int) or explicit state(str)')
        
    @property
    def get_qreg(self):
        return self._q

class ClassicalRegister(object):
    def __init__(self, cr):
        if isinstance(cr, int):
            self._c = np.zeros(cr)[np.newaxis].T
        elif isinstance(qr, str):
            if set(cr) <= set('01'):
                self._c = np.array(list(map(int,np.array(list(cr)))))[np.newaxis].T
            else:
                raise EntropicaError('ClassicalRegister', 'If specifying explicit state(str), please input sting of 1"s and 0"s')
        else:
            raise EntropicError('ClassicalRegister','Please input size(int) or explicit state(str)')
        
    @property
    def get_creg(self):
        return self._c

class Result(object):
    def get_statevector(circuit_name):
        return circuit_name.state_vec
    
    def get_PDM(circuit_name):
        pdm = np.asmatrix(Result.get_statevector(circuit_name)).dot(np.asmatrix(Result.get_statevector(circuit_name)).getH())
        return pdm
    
    def get_ZPV(circuit_name):
        zpv = np.diagonal(Result.get_PDM(circuit_name))[np.newaxis].T
        return zpv
