import tensorflow as tf
from operators import Operators
from qutils import Qutils

ops = Operators()
qutils = Qutils()
# from math import pi, sqrt

class States():
    """
    Lets keep the syntax similar to the ones used in Qutip to make it easier to compare 
    for example sigmap for sigma_plus etc 

    We used the convention of considering the coupled qubit-cavity state as a tensor product in the order (qubit, cavity)
    """
    def __init__(self):
        pass 

    def basis(self, N, n):
        if isinstance(n, int):
            obj = tf.one_hot(indices=n, depth=N, dtype=tf.complex64)
            obj = tf.reshape(obj, (1, N, 1))
        else:
            try:
                batch_size=n.shape[0]
            except:
                batch_size = 1
            obj = tf.one_hot(indices=n.numpy(), depth=N, dtype=tf.complex64)
            obj = tf.reshape(obj, (batch_size,N,1))
        return obj


    def vacuum(self, N, batch_size=1):
        vac = tf.zeros(batch_size)
        return self.basis(N, vac) 

    def ancilla_init(self, batch_size=1):
        ancilla = (self.basis(2, tf.zeros(batch_size)) 
                + self.basis(2, tf.ones(batch_size)))/tf.cast(tf.sqrt(2.0), tf.complex64)
        return tf.cast(qutils.ket2dm(ancilla), dtype=tf.complex64)


    def qubit_cavity_state(self, N, nq, nc):
        if isinstance(nq, int) and isinstance(nc, int):
            batch_size = 1
        else:
            try:
                batch_size = max([nq.shape[0], nc.shape[0]])
            except:
                batch_size = 1
        qubit_states = qutils.ket2dm(self.basis(2, nq))
        cavity_states = qutils.ket2dm(self.basis(N, nc))
        qubit_cavity_states = qutils.tensor([qubit_states, cavity_states])
        return qubit_cavity_states


    def coherent(self, N, alpha):
        D = ops.displace(N, alpha, tensored=False)
        batch_size = D.shape[0]
        cs = tf.reshape(tf.matmul(D, self.basis(N, tf.zeros(batch_size))), (batch_size, N, 1)) 
        return cs 


    def coherent_dm(self, N, alpha):
        return qutils.ket2dm(self.coherent(N, alpha))
        

    def fock(self, N, n):
        if isinstance(n, int):
            obj = tf.one_hot(indices=n, depth=N, dtype=tf.complex64)
            obj = tf.reshape(obj, (1, N, 1))
        else:
            try:
                batch_size=n.shape[0]
            except:
                batch_size = 1
            obj = tf.one_hot(indices=n.numpy(), depth=N, dtype=tf.complex64)
            obj = tf.reshape(obj, (batch_size,N,1))
        return obj 


    def fock_dm(self, N, n):
        return qutils.ket2dm(self.fock(N, n))


    def gkp(self, N, batch_size=1, alpha=None, beta=None, mu=0, L=None, M=None, Delta=None):
        if L == None: 
            L = int(tf.sqrt(N+0.0)-1)
        if M == None: 
            M = int(tf.sqrt(N+0.0)-1)
        k,l = tf.meshgrid(tf.range(L) - L//2, tf.range(M) - M//2)
        k = tf.cast(tf.reshape(k, [-1]), dtype=tf.complex64)
        l = tf.cast(tf.reshape(l, [-1]), dtype=tf.complex64)
        alpha = tf.cast(alpha, tf.complex64)
        beta = tf.cast(beta, tf.complex64)
        try:
            pi = tf.constant(pi, tf.complex64)
        except:
            from math import pi
            pi = tf.constant(pi, tf.complex64)
        mu = tf.constant(mu, tf.complex64)

        psi = sum(tf.cast(tf.exp(-1j*pi*(k*l+mu*l/2)), dtype=tf.complex64) * self.coherent(N,(2*k+mu)*alpha+l*beta) 
                for k, l in zip(k, l))
        if Delta != 0.0 or Delta is not None:
            Delta = tf.cast(Delta, tf.complex64)
            n = -Delta**2 * ops.num(N, batch_size=1)
            G = tf.linalg.expm(n)
            psi = tf.matmul(G, psi)
        norm = tf.math.sqrt(qutils.dag(psi) @ psi)
        psi = psi/norm
        psi = qutils.make_batch(psi, batch_size)
        return psi


    def gkp_dm(self, N, batch_size=1, alpha=None, beta=None, mu=0, L=None, M=None, Delta=None):
        psi = self.gkp(N, batch_size, alpha=alpha, beta=beta, mu=mu, L=L, M=M, Delta=Delta)
        return qutils.ket2dm(psi)


    def gkp_ideal(self, N, batch_size=1, alpha=None, beta=None, mu=0, L=None, M=None, Delta=0.0):
        psi = self.gkp(N, batch_size, alpha=alpha, beta=beta, mu=mu, L=L, M=M, Delta=0.0)
        return psi 


    def gkp_approx(self, N, batch_size=1, alpha=None, beta=None, mu=0, L=None, M=None, Delta=0.2):
        if Delta == 0:
            print("Delta must be nonzero. Using default value 0.2 instead")
            Delta = 0.2 
        psi = self.gkp(N, batch_size, alpha=alpha, beta=beta, mu=mu, L=L, M=M, Delta=Delta)
        return psi 


    def qubit_gkp_state(self, N, batch_size=1, alpha=None, beta=None, mu=0, L=None, M=None, Delta=None):
        qubit_state = qutils.unit(ops.identity(2, batch_size=batch_size))
        gkp_state = self.gkp_dm(N, batch_size=batch_size, alpha=alpha, beta=beta, mu=0, L=None, M=None, Delta=Delta)
        qubit_cavity = qutils.tensor([qubit_state, gkp_state])
        return qubit_cavity

