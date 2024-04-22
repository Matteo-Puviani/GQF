import tensorflow as tf
from math import pi, sqrt
from qutils import Qutils
from states import States
from operators import Operators 

ops = Operators()
qutils = Qutils()
states = States()

class MESolve:
    def __init__(self, N=100, batch_size=1, noise_level='low', Hamiltonian='off', chi=None, K_s=None):
    '''
    Class to solve the Lindblad master equation for the dissipative time-evolution. 

    PARAMETERS:

        N (int): Size of the cavity Fock space. Default value is 100. 

        batch_size (int): Number of parallel trajectories run to train a single NN agent. Default value is 5.

        noise_level (str): Strength of the dynamics' noise. Possible values are: 'low' (T_s, T_1, T_2 = 610, 280, 238), 'medium' (T_s, T_1, T_2 = 490, 100, 120) or 'high' (T_s, T_1, T_2 = 245, 50, 60). Default level is 'low'.
 
        Hamiltonian (str): If 'on' the Hamiltonian dynamics is also considered in the noisy time evolution. Possible values are 'on' and 'off'.

        chi (float): Value of the dissipative coupling between cavity and transmon. Specify a value only if the Hamiltonian is 'on' and a different value from the ones specified by the dynamics is desired. Default: None.

        K_s (float): Value of the Kerr nonlinearity. Specify a value only if the Hamiltonian is 'on' and a different value from the ones specified by the dynamics is desired. Default: None.

    '''
        if 'Sivak2023' in Hamiltonian:
            chi, K_s =  2.*3.141592654*0.0465, 2.*3.141592654*4.8*1E-6
        elif 'Ibarcq2020' in Hamiltonian:
            chi, K_s =  2.*3.141592654*0.028, 2.*3.141592654*1E-6

        if noise_level == 'high':
            T_s, T_1, T_2 = 245, 50, 60
        elif noise_level == 'medium':
            T_s, T_1, T_2 = 490, 100, 120
        elif noise_level == 'low':
            T_s, T_1, T_2 = 610, 280, 238

        T_phi = 1./(1./T_2 - 0.5/T_1)
       
        if Hamiltonian is not 'off':
            Coupl = chi/2. * qutils.tensor([ops.sigmaz(N=2, batch_size=batch_size),tf.matmul(ops.create(N, batch_size=batch_size),ops.destroy(N, batch_size=batch_size))])
            Kerr = K_s*tf.square(tf.matmul(ops.create(N, tensored=True, batch_size=batch_size),ops.destroy(N, tensored = True, batch_size=batch_size)))
        
        if Hamiltonian == 'off':
            self.Hamiltonian = tf.eye(2*N, dtype=tf.complex64)
        elif Hamiltonian == 'coupling':
            self.Hamiltonian = Coupl
        elif Hamiltonian == 'Kerr':
            self.Hamiltonian = Kerr
        elif Hamiltonian == 'on' or 'Sivak2023' in Hamiltonian or 'Ibarcq2020' in Hamiltonian:
            self.Hamiltonian = Coupl+Kerr
        else:
            raise(NotImplementedError)
        c_ops = [ops.destroy(N=N, tensored=True, batch_size=1)/sqrt(T_s),
                 ops.sigmap(N=N, tensored=True, batch_size=1)/sqrt(T_1),
                 ops.sigmaz(N=N, tensored=True, batch_size=1)/sqrt(2.*T_phi)]
        
        e_ops = []    

        self.c_ops0 = qutils.make_batch(c_ops[0], batch_size)
        self.c_ops1 = qutils.make_batch(c_ops[1], batch_size)
        self.c_ops2 = qutils.make_batch(c_ops[2], batch_size)

        self.Ad0 = qutils.dag(self.c_ops0)
        self.Ad1 = qutils.dag(self.c_ops1)
        self.Ad2 = qutils.dag(self.c_ops2)

        self.AdA0 = tf.matmul(self.Ad0, self.c_ops0) 
        self.AdA1 = tf.matmul(self.Ad1, self.c_ops1)
        self.AdA2 = tf.matmul(self.Ad2, self.c_ops2)


    def f_fast(self, rhot):
        comm = tf.matmul(self.Hamiltonian, rhot)  - tf.matmul(rhot, self.Hamiltonian) 
        C0 = qutils.matmul3(self.c_ops0, rhot, self.Ad0) - 0.5 * (tf.matmul(rhot, self.AdA0) + tf.matmul(self.AdA0, rhot))
        C1 = qutils.matmul3(self.c_ops1, rhot, self.Ad1) - 0.5 * (tf.matmul(rhot, self.AdA1) + tf.matmul(self.AdA1, rhot))
        C2 = qutils.matmul3(self.c_ops2, rhot, self.Ad2) - 0.5 * (tf.matmul(rhot, self.AdA2) + tf.matmul(self.AdA2, rhot))
        return -1j*comm + C0 + C1 + C2  

    def RK4(self, rho0, time_steps=10, tmax=1.1):
        i = tf.constant(0)
        rhot = rho0
        dt = tf.cast(tmax/time_steps, tf.complex64)

        def loop_cond(i, rhot):
            return tf.less(i, time_steps)

        def loop_body(i, rhot):
            k1 = dt * self.f_fast(rhot)
            k2 = dt * self.f_fast(rhot + k1/2)
            k3 = dt * self.f_fast(rhot + k2/2)
            k4 = dt * self.f_fast(rhot + k3)
            rhot += (k1 + 2*k2 + 2*k3 + k4)/6
            return [tf.add(i, 1), rhot] 

        i, rhot = tf.while_loop(loop_cond, loop_body, [i, rhot])
        return rhot
    
