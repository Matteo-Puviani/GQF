import tensorflow as tf

class Qutils():
    """
    Class of utility functions and operators. 
    We used the convention of considering the coupled qubit-cavity state as a tensor product in the order (qubit, cavity).
    """
    def __init__(self):
        pass 


    def Rxy(self, theta, phi, Lambda):
        """
        theta, phi, Lambda need to be np arrays
        """
        batch_size = theta.shape[0] 
        theta=tf.reshape(tf.cast(tf.math.real(theta), dtype=tf.float32), (-1))
        phi=tf.reshape(tf.cast(tf.math.real(phi), dtype=tf.float32), (-1))
        Lambda=tf.reshape(tf.cast(tf.math.real(Lambda), dtype=tf.float32), (-1))
        rxy = tf.Variable(tf.zeros((batch_size,2,2),dtype=tf.complex64))
        for ii in range(batch_size):
            rxy[ii,0,0].assign(tf.cast(tf.cos(theta[ii]/2), dtype=tf.complex64))
            rxy[ii,0,1].assign(tf.cast(tf.math.exp(1j * tf.cast(phi[ii],dtype=tf.complex64)), dtype=tf.complex64) * tf.cast(tf.sin(theta[ii]/2), dtype=tf.complex64))
            rxy[ii,1,0].assign(- tf.cast(tf.math.exp(1j * tf.cast(Lambda[ii],dtype=tf.complex64)), dtype=tf.complex64) * tf.cast(tf.sin(theta[ii]/2), dtype=tf.complex64))
            rxy[ii,1,1].assign(tf.cast(tf.math.exp(1j * tf.cast((phi[ii] + Lambda[ii]),dtype=tf.complex64)), dtype=tf.complex64) * tf.cast(tf.cos(theta[ii]/2), dtype=tf.complex64))
        return rxy


    def identity(self, N, batch_size=1):
        identity =  tf.eye(N, dtype=tf.complex64)
        return tf.tile(tf.expand_dims(identity, axis=0), multiples=[batch_size, 1, 1]) 


    def make_batch(self, array, batch_size):
        array = tf.squeeze(array, 0)
        dataset = tf.data.Dataset.from_tensor_slices([array]).repeat(batch_size)
        batched_dataset = dataset.batch(batch_size)
        array = tf.concat(list(batched_dataset.as_numpy_iterator()), axis=0)
        return array


    def tr(self, rho):
        batch_size = rho.shape[0]
        return tf.reshape(tf.linalg.trace(rho), (batch_size,1,1))


    def unit(self, rho):
        batch_size=rho.shape[0]
        rho /= tf.reshape(tf.linalg.trace(rho), (batch_size,1,1))
        return rho


    def dag(self, Q):
        return tf.linalg.matrix_transpose(tf.math.conj(Q))


    def ket2dm(self, Q): #works only for shape (batch, N, 1), while for (batch, N) you need to invert Q and Q.dag
        return self.unit(tf.linalg.matmul(Q, tf.linalg.matrix_transpose(tf.math.conj(Q))))


    def tensor(self, rhos):
        operators = list(map(tf.linalg.LinearOperatorFullMatrix, rhos))
        tensor_state = tf.linalg.LinearOperatorKronecker(operators).to_dense()
        return tensor_state


    def ptrace(self, rho, axis, dim1=2, dim2=100, batch_size=None):
        if batch_size is None:
            batch_size = int(rho.shape[0])
        if axis==0:
            rho_partial = tf.linalg.einsum("bikjk->bij", tf.reshape(rho, (batch_size, dim1, dim2, dim1, dim2)))
        else:
            rho_partial =  tf.linalg.einsum("bkikj->bij", tf.reshape(rho, (batch_size, dim1, dim2, dim1, dim2)))
        return rho_partial


    def matmul3(self, A, B, C):
        return tf.einsum('bij,bjk,bkl->bil', A, B, C)


    def reset_ancilla(self, rho, target_state=None):
        N = int(rho.shape[1]/2)
        bosonic = self.ptrace(rho=rho, axis=1, dim1=2, dim2=N)
        rho_reset = self.tensor([target_state, bosonic])
        return rho_reset 


    def fidelity(self, rho, rho_target, batch_size=1):
        rho = self.ptrace(rho, axis=1, dim1=2, dim2=self.N, batch_size=self.batch_size) 
        s_sqrt = self.matrix_exp(rho_target)
        prod = tf.linalg.matmul(s_sqrt, tf.linalg.matmul(rho, s_sqrt))
        prod_sqrt = self.matrix_exp(prod)
        trace = tf.einsum('bii->b', prod_sqrt)
        f = tf.math.real(trace)   
        return f.numpy()
        

