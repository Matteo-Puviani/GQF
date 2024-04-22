import tensorflow as tf
from qutils import Qutils

qutils = Qutils()

class Operators():
    """
    Class of the operators acting on the cavity or/and on the ancilla qubit.
    We used the convention of considering the coupled qubit-cavity state as a tensor product in the order (qubit, cavity).
    Some operators can be tensored (acting on both the cavity and the qubit).
    """
    def __init__(self):
        pass 


    def identity(self, N, batch_size=1):
        identity =  tf.eye(N, dtype=tf.complex64)
        return tf.tile(tf.expand_dims(identity, axis=0), multiples=[batch_size, 1, 1]) 


    def destroy(self, N, tensored=False, batch_size=1):
        a = tf.tile(tf.expand_dims(tf.cast(tf.linalg.diag(tf.sqrt(tf.range(1, N, dtype=tf.float32)), k=1), 
            dtype=tf.complex64) , axis=0), multiples=[batch_size, 1, 1]) 
        if tensored:
            Id = self.identity(2, batch_size)
            a = qutils.tensor([Id, a])
        return a 


    def create(self, N, tensored=False, batch_size=1):
        ad = qutils.dag(self.destroy(N, tensored, batch_size))
        return ad 


    def num(self, N, tensored=False, batch_size=1):
        n = tf.tile(tf.expand_dims(tf.cast(tf.linalg.diag(tf.range(0, N)), dtype=tf.complex64), axis=0), 
                multiples=[batch_size, 1, 1])
        if tensored:
            Id = self.identity(2, batch_size)
            n = qutils.tensor([Id, n])
        return n    


    def sigmax(self, N=5, tensored=False, batch_size=1): 
        if tensored:
            sigmax = qutils.tensor([tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.complex64), 
                tf.eye(N, dtype=tf.complex64)])
        else:
            sigmax = tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=tf.complex64)
        return tf.tile(tf.expand_dims(sigmax, axis=0), multiples=[batch_size, 1, 1])


    def sigmay(self, N=5, tensored=False, batch_size=1):
        if tensored:
            sigmay = qutils.tensor([tf.constant([[0.0j, -1.0j], [1.0j, 0.0j]], dtype=tf.complex64), 
                tf.eye(N,dtype=tf.complex64)])
        else:
            sigmay = tf.constant([[0.0j, -1.0j], [1.0j, 0.0j]], dtype=tf.complex64)
        return tf.tile(tf.expand_dims(sigmay, axis=0), multiples=[batch_size, 1, 1])


    def sigmaz(self, N=5, tensored=False, batch_size=1):
        if tensored:
            sigmaz = qutils.tensor([tf.constant([[1.0, 0.0], [0.0, -1.0]], dtype=tf.complex64),tf.eye(N,dtype=tf.complex64)])
        else:
            sigmaz = tf.constant([[1.0, 0.0], [0.0, -1.0]], dtype=tf.complex64)
        return tf.tile(tf.expand_dims(sigmaz, axis=0), multiples=[batch_size, 1, 1])


    def sigmam(self, N=5, tensored=False, batch_size=1):
        if tensored: 
            sigmam = qutils.tensor([tf.constant([[0.0, 1.0], [0.0, 0.0]], dtype=tf.complex64),tf.eye(N,dtype=tf.complex64)])
        else:
            sigmam = tf.constant([[0.0, 1.0], [0.0, 0.0]], dtype=tf.complex64) 
        return tf.tile(tf.expand_dims(sigmam, axis=0), multiples=[batch_size, 1, 1])


    def sigmap(self, N=5, tensored=False, batch_size=1):
        if tensored:
            sigmap = qutils.tensor([tf.constant([[0.0, 0.0], [1.0, 0.0]], dtype=tf.complex64),tf.eye(N,dtype=tf.complex64)])
        else:
            sigmap = tf.constant([[0.0, 0.0], [1.0, 0.0]], dtype=tf.complex64)
        return tf.tile(tf.expand_dims(sigmap, axis=0), multiples=[batch_size, 1, 1]) 


    def displace(self, N, alpha, batch_size=1, tensored=False):
        if batch_size ==1:
            alpha = tf.constant(alpha, dtype=tf.complex64) 
        a = self.destroy(N=N, batch_size=batch_size)
        ad = qutils.dag(a)
        alpha = tf.reshape(alpha, (batch_size, 1, 1))
        D = tf.linalg.expm(alpha * ad - tf.math.conj(alpha) * a)
        if tensored:
            Id = self.identity(N=2, batch_size=batch_size)
            D = qutils.tensor([Id, D])
        return D


    def cond_displace(self, N, alpha, batch_size=1):
        if batch_size ==1:
            alpha = tf.constant(alpha, dtype=tf.complex64)
        
        a = self.destroy(N=N, batch_size=batch_size)
        ad = qutils.dag(a)
        D = alpha * ad - tf.math.conj(alpha) * a
        CD = tf.linalg.expm(qutils.tensor([0.5 * self.sigmaz(N=N,batch_size=batch_size), D]))
        return CD


    def feedback_displace(self, N, alpha, batch_size=1, tensored=True):
        D = self.displace(N, alpha, batch_size=batch_size, tensored=tensored)
        return D


    def POVM(self, rho, theta, phi, Lambda, deterministic=False, mu=None):
        batch_size = rho.shape[0]
        N = int(rho.shape[1]/2)

        direction = qutils.Rxy(theta=theta, phi=phi, Lambda=Lambda)
        values, vectors = tf.linalg.eig(direction)
        ground = tf.math.argmin(tf.math.real(values), axis=1)
        state_plus = tf.gather(vectors, ground, batch_dims=1, axis=1)
        state_plus = qutils.ket2dm(tf.reshape(state_plus, (batch_size,2,1)))
        excited = 1 - ground
        state_minus = tf.gather(vectors, excited, batch_dims=1, axis=1)
        state_minus = qutils.ket2dm(tf.reshape(state_minus,( batch_size,2,1)))
        Proj = qutils.tensor([state_plus, self.identity(N, batch_size=batch_size)])
        probability = tf.math.real(tf.linalg.trace(qutils.matmul3(Proj,rho, qutils.dag(Proj))))
        if deterministic:
            if mu != None:
                mu = tf.reshape(tf.cast(mu, tf.complex64),(batch_size,1,1))
                num_mask = tf.ones((batch_size,1,1), tf.complex64)-mu
                vector_masked = num_mask*state_plus+(1-num_mask)*state_minus
                Proj = qutils.tensor([vector_masked,self.identity(N, batch_size=batch_size)])
            else:
                mask = ((probability-0.5)*2) > 0
                num_mask = tf.reshape(tf.cast(mask, dtype=tf.complex64), (batch_size,1,1))
                vector_masked = num_mask*state_plus+(1-num_mask)*state_minus
                Proj = qutils.tensor([vector_masked,self.identity(N, batch_size=batch_size)])
        else:
            ppp = tf.random.uniform(shape=[batch_size]).numpy()
            mask = ppp < probability
            num_mask = tf.reshape(tf.cast(mask, dtype=tf.complex64), (batch_size,1,1))
            vector_masked = num_mask*state_plus+(1-num_mask)*state_minus
            Proj = qutils.tensor([vector_masked, self.identity(N, batch_size=batch_size)])
        prob = tf.cast(num_mask, dtype=tf.float32)*tf.reshape(probability,
                (batch_size,1,1))+(1.-tf.cast(num_mask, dtype=tf.float32))*(1.-tf.reshape(probability,
                (batch_size,1,1)))

        rho = qutils.matmul3(Proj, rho, qutils.dag(Proj))/tf.cast(prob, dtype=tf.complex64)
        sign = tf.cast((num_mask*2.)-1., dtype=tf.complex64)
        
        return qutils.unit(rho), sign, prob



    def gkp_Pauli_Z(self, N, alpha=None, beta=None, Delta=None, tensored=True, batch_size=1):
        if not tensored:
            PZ = self.displace(N, beta, False)
        else:
            PZ = qutils.tensor([tf.eye(2, dtype=tf.complex64), self.displace(N, beta, 1, False)])
            PZ = tf.reshape(PZ,(2*N,2*N))
        return tf.tile(tf.expand_dims(PZ, axis=0), multiples=[batch_size, 1, 1])

    def gkp_Pauli_Sz(self, N, alpha=None, beta=None, Delta=None, tensored=True, batch_size=1):
        if not tensored:
            PZ = self.displace(N, 2*beta, False)
        else:
            PZ = qutils.tensor([tf.eye(2, dtype=tf.complex64), self.displace(N, 2*beta, 1, False)])
            PZ = tf.reshape(PZ, (2*N,2*N))
        return tf.tile(tf.expand_dims(PZ, axis=0), multiples=[batch_size, 1, 1])

    def gkp_Pauli_X(self, N, alpha=None, beta=None, Delta=None, tensored=True, batch_size=1):
        if not tensored:
            PX = self.displace(N, alpha, False)
        else:
            PX = qutils.tensor([tf.eye(2, dtype=tf.complex64), self.displace(N, alpha, 1, False)])
            PX = tf.reshape(PX,(2*N,2*N))
        return tf.tile(tf.expand_dims(PX, axis=0), multiples=[batch_size, 1, 1])

    def gkp_Pauli_Y(self, N, alpha=None, beta=None, Delta=None, tensored=True, batch_size=1):
        if not tensored:
            PY = self.displace(N, alpha+beta, False)
        else:
            PY = qutils.tensor([tf.eye(2, dtype=tf.complex64), self.displace(N, alpha+beta, 1, False)])
            PY = tf.reshape(PY,(2*N,2*N))
        return tf.tile(tf.expand_dims(PY, axis=0), multiples=[batch_size, 1, 1])


    def gkp_Pauli_Sx(self, N, alpha=None, beta=None, Delta=None, tensored=True, batch_size=1):
        if not tensored:
            PX = self.displace(N, 2*alpha, False)
        else:
            PX = qutils.tensor([tf.eye(2, dtype=tf.complex64), self.displace(N, 2*alpha, 1, False)])
            PX = tf.reshape(PX,(2*N,2*N))
        return tf.tile(tf.expand_dims(PX, axis=0), multiples=[batch_size, 1, 1])


    def qubit_rotation(self, N, phi, theta, batch_size=1):
        if batch_size ==1:
            phi = tf.constant(phi, dtype=tf.complex64)
            theta = tf.constant(theta, dtype=tf.complex64)
        exponent = self.sigmax(batch_size=batch_size)*tf.cos(phi)+self.sigmay(batch_size=batch_size)*tf.sin(phi)
        R = qutils.tensor([tf.linalg.expm(-1j*theta*0.5*exponent),self.identity(N=N, batch_size=batch_size)])
        return tf.cast(R, tf.complex64)


    def echoed_cond_displace(self, N, alpha, batch_size=1):
        if batch_size ==1:
            alpha = tf.constant(alpha, dtype=tf.complex64)
        
        a = self.destroy(N=N, batch_size=batch_size)
        ad = qutils.dag(a)
        D_plus = tf.linalg.expm(alpha * ad/2. - tf.math.conj(alpha) * a/2.)
        D_minus = tf.linalg.expm(- alpha * ad/2. + tf.math.conj(alpha) * a/2.)
        ECD = qutils.tensor([self.sigmam(batch_size=batch_size), D_plus])+qutils.tensor([self.sigmap(batch_size=batch_size), D_minus])
        return ECD


    def cavity_rotation(self, N, theta, batch_size=1):
        theta = tf.constant(theta, dtype=tf.complex64)
        
        a = self.destroy(N=N, batch_size=batch_size)
        ad = qutils.dag(a)

        exponent = 1j*theta*tf.matmul(ad,a)
        VR =  qutils.tensor([self.identity(N=2,batch_size=batch_size),tf.linalg.expm(exponent)])

        return tf.cast(VR, tf.complex64)
