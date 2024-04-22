import gym
import numpy as np
from gym import spaces

import tensorflow as tf
from math import pi, sqrt

from states import States
from operators import Operators
from qutils import Qutils
from mesolve import MESolve

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

states = States()
ops = Operators()
qutils = Qutils()


class GKPEnv(gym.Env):

    def __init__(self, 
        N=100,
        Delta=0.20, 
        noise_level='low',
    	max_steps=21,
    	memory_length=1,
        protocol='sBs',
        initialized_values=True,
        batch_size=5, 
    	dynamics=True,
        deterministic=False,
        mu=None,
        dynamics_type='Sivak2023',
        Hamiltonian='off', 
        chi=None, 
        K_s=None,
        track_interval=1,
    	save_interval=10, 
        reward_type=None,
    	evaluate=False, 
        initial_state='zero logical',
        verbosity = 'low',
        TEST=False
    	):

        """
        PARAMETERS
        
        N (int): Size of the cavity Fock space. Default value is 100. 
        
        Delta (float): Delta value of the GKP code. Default value is 0.2

        noise_level (str): Strength of the dynamics' noise. Possible values are: 'low' (T_s, T_1, T_2 = 610, 280, 238), 'medium' (T_s, T_1, T_2 = 490, 100, 120) or 'high' (T_s, T_1, T_2 = 245, 50, 60). Default level is 'low'.

        max_steps (int): Maximum number of half-cycles of each trajectory, suggested to be an odd number as the QEC cycle ends at the beginning of the following step. Default value is 21.  

        memory_length (int): Number of measurements to provide as input to the neural network. Default value is 1 (and memory is achieved anyway using a recurrent network).

        protocol (str): QEC protocol to be used. Default: 'sBs'. No other protocols implemented for now.

        initialized_values (bool): If yes, the gates' parameters are initialized according to the 'best' theoretical constant values, and the neural network will apply corrections to those.

        batch_size (int): Number of parallel trajectories run to train a single NN agent. Default value is 5. 

        dynamics (bool): If 'True' the dissipative dynamics specified in 'dynamics_type' is considered in the simulation.

        deterministic (bool, optional): If 'yes' the measurement outputs are the ones given by 'mu'.

        mu (int, array): Sequence of measurement outputs to be postselected, if deterministic is 'True'.

        dynamics_type (str): It allows to implement some specific time-duration sequence of the noisy dynamics. Possible values are 'Sivak2023', 'Ibarcq2020', 'autonomous', 'simple'.

        Hamiltonian (str): If 'on' the Hamiltonian dynamics is also considered in the noisy time evolution. Possible values are 'on' and 'off'.

        chi (float, opt): Value of the dissipative coupling between cavity and transmon. Specify a value only if the Hamiltonian is 'on' and a different value from the ones specified by the dynamics is desired. Default: None.

        K_s (float, opt): Value of the Kerr nonlinearity. Specify a value only if the Hamiltonian is 'on' and a different value from the ones specified by the dynamics is desired. Default: None.

        track_interval (int): Number of trajectories after which we want to track the values.

        save_interval (int): Number of trajectories after which we want to save the agent.

        reward_type (str): Type of reward for the neural network. Possible values are: 'Pauli Z', 'fidelity', 'Pauli Z + fidelity'.

        evaluate (bool): If 'True' it evaluates the NN agent loaded. 

        initial_state (str, list(complex), optional): Initial state for the simulation. Possible values are: 'zero logical', 'one logical', 'x logical', '-x logical', 'y logical', '-y logical'. Alternatively it is possible to give a list of two complex numbers (a,b) such that the initial state is |i > = a |0_L > + b | 1_L >, with a^2 + b^2 = 1. Default: 'zero logical'

        verbosity (str):  Possible values are: 'low', 'actions' (only the actions suggested by the NN), 'high', 'all'.

        TEST (bool, optional): Performing an evaluation test with constant parameters. Default: False 
        """

        super().__init__()

        self.N = N
        self.max_steps = max_steps
        self.memory_length = memory_length
        self.protocol = protocol
        self.initialized_values = initialized_values
        self.batch_size = batch_size
        self.dynamics = dynamics
        self.Hamiltonian = Hamiltonian
        self.chi = chi
        self.K_s = K_s
        self.track_interval = track_interval
        self.save_interval = save_interval
        self.evaluate = evaluate
        self.reward_type=reward_type
        if self.evaluate:
            self.save_interval = 1 

        self.dynamics_type = dynamics_type
        self.initial_state = initial_state
        self.deterministic = deterministic
        self.mu = mu
        self.verbosity = verbosity
        self.noise_level = noise_level
        self.TEST = TEST

        ## parameters and variables
        self.Delta = Delta
        self.alpha = qutils.make_batch(tf.reshape(tf.constant(tf.sqrt(tf.cast(pi/2, dtype=tf.complex64)), dtype=tf.complex64),
             (-1, 1, 1)), self.batch_size)
        self.beta = qutils.make_batch(tf.reshape(tf.constant(1j*tf.sqrt(tf.cast(pi/2, dtype=tf.complex64)), dtype=tf.complex64),
             (-1, 1, 1 )), self.batch_size)
        self.theta = qutils.make_batch(tf.reshape(tf.Variable(tf.cast(0., dtype=tf.complex64), dtype=tf.complex64), (-1, 1, 1 )), self.batch_size)
        self.phi = qutils.make_batch(tf.reshape(tf.Variable(0.0, dtype=tf.complex64), (-1, 1, 1 )), self.batch_size)
        self.Lambda = qutils.make_batch(tf.reshape(tf.Variable(tf.cast(0., dtype=tf.complex64), dtype=tf.complex64), (-1, 1, 1)), self.batch_size)

        self.ancilla_init = qutils.make_batch(qutils.ket2dm(states.basis(N=2,n=0)), self.batch_size)

        if 'mixed' in self.Hamiltonian:
            solver = MESolve(N=self.N, batch_size=self.batch_size, noise_level=self.noise_level, Hamiltonian='off', chi=self.chi, K_s=self.K_s)
            self.RK4_fast = tf.function(solver.RK4)
            
            solver_disp = MESolve(N=self.N, batch_size=self.batch_size, noise_level=self.noise_level, Hamiltonian=self.Hamiltonian, chi=self.chi, K_s=self.K_s)
            self.RK4_fast_disp = tf.function(solver_disp.RK4)
        else:
            solver = MESolve(N=self.N, batch_size=self.batch_size, noise_level=self.noise_level, Hamiltonian=self.Hamiltonian, chi=self.chi, K_s=self.K_s)
            self.RK4_fast = tf.function(solver.RK4)



        self.vac = qutils.ket2dm(states.vacuum(self.N, self.batch_size))
        logical_zero = states.gkp(self.N, alpha=tf.reshape(self.alpha[0], (1, 1, 1)), 
            beta=tf.reshape(self.beta[0], (1, 1, 1)), mu=0, Delta=self.Delta, batch_size=1)
        logical_one = states.gkp(self.N, alpha=tf.reshape(self.alpha[0], (1, 1, 1)), 
            beta=tf.reshape(self.beta[0], (1, 1, 1)), mu=1, Delta=self.Delta, batch_size=1)

        self.theta_0 = tf.reshape(tf.repeat(tf.cast(pi/2., tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))

        if self.protocol == 'sBs_bias':
            self.phi1_0 = tf.reshape(tf.repeat(tf.cast(pi/2.+0.05, tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.phi2_0 = tf.reshape(tf.repeat(tf.cast(0.-0.03,tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.phi3_0 = tf.reshape(tf.repeat(tf.cast(0.-0.06,tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.phi4_0 = tf.reshape(tf.repeat(tf.cast(pi/2.+0.04, tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))

            self.theta1_0 = tf.reshape(tf.repeat(tf.cast(pi/2.-0.03, tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.theta2_0 = tf.reshape(tf.repeat(-tf.cast(pi/2.+0.05, tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.theta3_0 = tf.reshape(tf.repeat(tf.cast(pi/2.+0.06, tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.theta4_0 = tf.reshape(tf.repeat(-tf.cast(pi/2.+0.04, tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))

            self.beta1_0 = tf.reshape(tf.repeat(tf.cast(1j*0.2+0.06-0.04*1j, tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.beta2_0 = tf.reshape(tf.repeat(tf.cast(sqrt(2.*pi)+0.04-0.02*1j, tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.beta3_0 = tf.reshape(tf.repeat(tf.cast(1j*0.2+0.04-0.05*1j, tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.beta4_0 = tf.reshape(tf.repeat(tf.cast(0.,tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
    
        else:
            self.phi1_0 = tf.reshape(tf.repeat(tf.cast(pi/2., tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.phi2_0 = tf.reshape(tf.repeat(tf.cast(0.,tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.phi3_0 = tf.reshape(tf.repeat(tf.cast(0.,tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.phi4_0 = tf.reshape(tf.repeat(tf.cast(pi/2., tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))

            self.theta1_0 = tf.reshape(tf.repeat(tf.cast(pi/2., tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.theta2_0 = tf.reshape(tf.repeat(-tf.cast(pi/2., tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.theta3_0 = tf.reshape(tf.repeat(tf.cast(pi/2., tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.theta4_0 = tf.reshape(tf.repeat(-tf.cast(pi/2., tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))

            self.beta1_0 = tf.reshape(tf.repeat(tf.cast(1j*0.2, tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.beta2_0 = tf.reshape(tf.repeat(tf.cast(sqrt(2.*pi), tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.beta3_0 = tf.reshape(tf.repeat(tf.cast(1j*0.2, tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
            self.beta4_0 = tf.reshape(tf.repeat(tf.cast(0.,tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))


        if isinstance(self.initial_state,str):
            if self.initial_state == 'zero logical':
                a = 1.0
                b = 0.
            elif self.initial_state == 'one logical':
                a = 0.
                b = 1.0
            elif self.initial_state == 'x logical':
                a = 1./sqrt(2.)
                b = 1./sqrt(2.)
            elif self.initial_state == '-x logical':
                a = 1./sqrt(2.) 
                b = -1./sqrt(2.)
            elif self.initial_state == 'y logical':
                a = 1./sqrt(2.)
                b = 1j*1./sqrt(2.)
            elif self.initial_state == '-y logical':
                a = 1./sqrt(2.)
                b = -1j*1./sqrt(2.)
            else:
                raise(ValueError)
        elif isinstance(self.initial_state, list):
            a = self.initial_state[0]
            b = self.initial_state[1]
            if tf.math.abs(a**2.+b**2.-1.) > 1E-4:
                print('ERROR: The initial state is not normalized.')
                raise(ValueError)
        else:
            raise(ValueError)

        logical_state = a * logical_zero + b * logical_one
        rho_target = qutils.ket2dm(logical_state)
        self.rho_target =  qutils.make_batch(rho_target, batch_size=self.batch_size) 
        self.rho0 = qutils.tensor([self.ancilla_init, qutils.make_batch(rho_target, self.batch_size)]) # QEC

        ## Stabilizers and Pauli operators
        self.Sz = ops.gkp_Pauli_Sz(N=self.N, 
               beta=tf.reshape(self.beta[0], (1, 1, 1)), Delta=0*self.Delta, batch_size=self.batch_size, tensored=True)
        self.PauliZ = ops.gkp_Pauli_Z(N=self.N,  
               beta=tf.reshape(self.beta[0], (1, 1, 1)), Delta=0*self.Delta, batch_size=self.batch_size, tensored=True)
        self.Sx = ops.gkp_Pauli_Sx(N=self.N, 
             alpha=tf.reshape(self.alpha[0], (1, 1, 1)), Delta=0*self.Delta, batch_size=self.batch_size, tensored=True)
        self.PauliX = ops.gkp_Pauli_X(N=self.N, 
             alpha=tf.reshape(self.alpha[0], (1, 1, 1)), Delta=0*self.Delta, batch_size=self.batch_size, tensored=True)
        self.PauliY = ops.gkp_Pauli_Y(N=self.N,
             alpha=tf.reshape(self.alpha[0], (1, 1, 1)), beta=tf.reshape(self.beta[0], (1, 1, 1)), Delta=0*self.Delta, batch_size=self.batch_size, tensored=True)

        # environment specific
        self.observation0 = tf.zeros(shape=int(self.batch_size * self.memory_length), dtype=tf.int32).numpy()
        self.episode_count = 0 
 
        # keep track 
        self.track_pauli = []
        self.track_pauliX = []
        self.track_pauliZ = []
        self.track_pauliY = []
        self.track_stabilizer = []
        self.track_stabilizerX = []
        self.track_stabilizerZ = []
        self.track_fidelity = []
        self.track_returns = []
        self.track_parameters = []
        self.track_meas = []
        self.track_prob = []
        self.track_rho = []

        if 'fidelity' in self.reward_type:
            if 'Pauli Z' in self.reward_type:
                _, self.R0, _ = self.pauli(self.rho0)
                self.R0 += self.compute_fidelity(self.rho0)
            else:
                self.R0 = self.compute_fidelity(self.rho0)
        elif 'Pauli Z' in self.reward_type and 'fidelity' not in self.reward_type:
            _, self.R0, _ = self.pauli(self.rho0)
        else:
            self.R0 = qutils.make_batch(tf.reshape(tf.cast(0., dtype=tf.float32),(-1, 1, 1)), self.batch_size)


    def QEC_sBs(self, actions=None, rho=None):

        theta = self.theta 
        phi = self.phi 
        Lambda = self.Lambda

        if self.timestep !=1:
            if self.dynamics_type == 'Sivak2023':
                if 'mixed' in self.Hamiltonian:
                    rho = self.RK4_fast_disp(rho, time_steps=20, tmax=2.3)
                else:
                    rho = self.RK4_fast(rho, time_steps=20, tmax=2.3)
            elif self.dynamics_type == 'autonomous':
                rho = self.RK4_fast(rho, time_steps=12, tmax=0.8)
            rho = qutils.reset_ancilla(rho=rho, target_state=qutils.ket2dm(qutils.make_batch(states.basis(N=2,n=0),batch_size=self.batch_size)))
        
            if self.TEST:
                if 'mixed' in self.Hamiltonian:
                    if self.dynamics_type == 'Sivak2023':
                        theta_VR = self.theta_0 - tf.reshape(tf.reshape(self.sign,(self.batch_size))*tf.repeat(tf.cast(2.3*2.*46.5*pi*0.001/2., tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1)) 
                    elif self.dynamics_type == 'Ibarcq2020':
                        theta_VR = self.theta_0 - tf.reshape(tf.reshape(self.sign,(self.batch_size))*tf.repeat(tf.cast(2.3*2.*28.*pi*0.001/2., tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))
                else:
                    theta_VR = self.theta_0
            else:
                if self.initialized_values:
                    if 'mixed' in self.Hamiltonian:
                        if self.dynamics_type == 'Sivak2023':
                            theta_VR = self.theta_0 - tf.reshape(tf.reshape(self.sign,(self.batch_size))*tf.repeat(tf.cast(2.3*2.*46.5*pi*0.001/2., tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))+tf.reshape(1.*actions[...,0], (self.batch_size, 1, 1))
                        elif self.dynamics_type == 'Ibarcq2020':
                            theta_VR = self.theta_0 - tf.reshape(tf.reshape(self.sign,(self.batch_size))*tf.repeat(tf.cast(2.3*2.*28.*pi*0.001/2., tf.complex64),repeats=self.batch_size,axis=0),(self.batch_size, 1, 1))+tf.reshape(1.*actions[...,0], (self.batch_size, 1, 1))
                    else:
                        theta_VR = self.theta_0+tf.reshape(1.*actions[...,0], (self.batch_size, 1, 1))
                else: 
                    theta_VR = tf.reshape(3.*actions[...,0], (self.batch_size, 1, 1))

            VR = ops.cavity_rotation(N=self.N,theta=theta_VR, batch_size=self.batch_size)  
            rho = qutils.matmul3(VR, rho, qutils.dag(VR))
            
            if self.dynamics_type == 'Sivak2023' or self.dynamics_type == 'autonomous':
                rho = self.RK4_fast(rho, time_steps=4, tmax=1.0)

            rho2=rho

        if self.TEST:
            phi1 = self.phi1_0
            phi2 = self.phi2_0
            phi3 = self.phi3_0
            phi4 = self.phi4_0

            theta1 = self.theta1_0
            theta2 = self.theta2_0
            theta3 = self.theta3_0
            theta4 = self.theta4_0

            beta1 = self.beta1_0
            beta2 = self.beta2_0
            beta3 = self.beta3_0
        else:

            if self.initialized_values: 
                phi1 = self.phi1_0 + tf.reshape(2. * actions[...,1], (self.batch_size, 1, 1))
                phi2 = self.phi2_0 + tf.reshape(2. * actions[...,2], (self.batch_size, 1, 1))
                phi3 = self.phi3_0 + tf.reshape(2. * actions[...,3], (self.batch_size, 1, 1))
                phi4 = self.phi4_0 + tf.reshape(2. * actions[...,4], (self.batch_size, 1, 1))

                theta1 = self.theta1_0 + tf.reshape(2. * actions[...,5], (self.batch_size, 1, 1))
                theta2 = self.theta2_0 + tf.reshape(2. * actions[...,6], (self.batch_size, 1, 1))
                theta3 = self.theta3_0 + tf.reshape(2. * actions[...,7], (self.batch_size, 1, 1))
                theta4 = self.theta4_0 + tf.reshape(2. * actions[...,8], (self.batch_size, 1, 1))

                beta1_r = tf.reshape(2. * actions[...,9], (self.batch_size, 1, 1))
                beta2_r = tf.reshape(2. * actions[...,10], (self.batch_size, 1, 1))
                beta3_r = tf.reshape(2. * actions[...,11], (self.batch_size, 1, 1))
                beta1_i= tf.reshape(2. * actions[...,12], (self.batch_size, 1, 1))
                beta2_i = tf.reshape(2. * actions[...,13], (self.batch_size, 1, 1))
                beta3_i = tf.reshape(2. * actions[...,14], (self.batch_size, 1, 1))

                beta1 = self.beta1_0 + beta1_r + 1j*beta1_i
                beta2 = self.beta2_0 + beta2_r + 1j*beta2_i
                beta3 = self.beta3_0 + beta3_r + 1j*beta3_i
            else:
                phi1 = tf.reshape(pi * actions[...,1], (self.batch_size, 1, 1))
                phi2 = tf.reshape(pi * actions[...,2], (self.batch_size, 1, 1))
                phi3 = tf.reshape(pi * actions[...,3], (self.batch_size, 1, 1))
                phi4 = tf.reshape(pi * actions[...,4], (self.batch_size, 1, 1))

                theta1 = tf.reshape(pi * actions[...,5], (self.batch_size, 1, 1))
                theta2 = tf.reshape(pi * actions[...,6], (self.batch_size, 1, 1))
                theta3 = tf.reshape(pi * actions[...,7], (self.batch_size, 1, 1))
                theta4 = tf.reshape(pi * actions[...,8], (self.batch_size, 1, 1))

                beta1_r = tf.reshape(2. * actions[...,9], (self.batch_size, 1, 1))
                beta2_r = tf.reshape(2. * actions[...,10], (self.batch_size, 1, 1))
                beta3_r = tf.reshape(2. * actions[...,11], (self.batch_size, 1, 1))
                beta1_i= tf.reshape(2. * actions[...,12], (self.batch_size, 1, 1))
                beta2_i = tf.reshape(2. * actions[...,13], (self.batch_size, 1, 1))
                beta3_i = tf.reshape(2. * actions[...,14], (self.batch_size, 1, 1))

                beta1 = beta1_r + 1j*beta1_i
                beta2 = beta2_r + 1j*beta2_i
                beta3 = beta3_r + 1j*beta3_i

        if self.dynamics_type == 'Sivak2023' or self.dynamics_type == 'autonomous':
            rho = self.RK4_fast(rho, time_steps=2, tmax=0.1)

        # layer 1
        rot = ops.qubit_rotation(N=self.N, phi=phi1, theta=theta1, batch_size=self.batch_size)
        rho = qutils.matmul3(rot, rho ,qutils.dag(rot))
        cond = ops.echoed_cond_displace(N=self.N, alpha=beta1, batch_size=self.batch_size)
        rho = qutils.matmul3(cond, rho ,qutils.dag(cond))
        
        if self.dynamics_type == 'Sivak2023' or self.dynamics_type == 'autonomous' or self.dynamics_type == 'simple':
            rho = self.RK4_fast(rho, time_steps=2, tmax=0.5) 

        # layer 2
        rot = ops.qubit_rotation(N=self.N, phi=phi2, theta=theta2, batch_size=self.batch_size)
        rho = qutils.matmul3(rot, rho ,qutils.dag(rot))
        cond = ops.echoed_cond_displace(N=self.N, alpha=beta2, batch_size=self.batch_size)
        rho = qutils.matmul3(cond, rho ,qutils.dag(cond))
        
        if self.dynamics_type == 'Sivak2023' or self.dynamics_type == 'autonomous':
            rho = self.RK4_fast(rho, time_steps=3, tmax=0.7) 
        elif self.dynamics_type == 'simple':
            rho = self.RK4_fast(rho, time_steps=2, tmax=0.5)

        # layer 3
        rot = ops.qubit_rotation(N=self.N, phi=phi3, theta=theta3, batch_size=self.batch_size)
        rho = qutils.matmul3(rot, rho ,qutils.dag(rot))
        cond = ops.echoed_cond_displace(N=self.N, alpha=beta3, batch_size=self.batch_size)
        rho = qutils.matmul3(cond, rho ,qutils.dag(cond))
        
        if self.dynamics_type == 'Sivak2023' or self.dynamics_type == 'autonomous':
            rho = self.RK4_fast(rho, time_steps=2, tmax=0.3) 
        elif self.dynamics_type == 'simple':
            rho = self.RK4_fast(rho, time_steps=2, tmax=0.5)

        # layer 4
        rot = ops.qubit_rotation(N=self.N, phi=phi4, theta=theta4, batch_size=self.batch_size)
        rho = qutils.matmul3(rot, rho ,qutils.dag(rot))
        displace = ops.displace(self.N, self.alpha, batch_size=self.batch_size, tensored=True) 
        rho = qutils.matmul3(displace, rho ,qutils.dag(displace))

        if self.dynamics_type == 'Sivak2023' or self.dynamics_type == 'autonomous':
            rho = self.RK4_fast(rho, time_steps=2, tmax=0.1) 
        elif self.dynamics_type == 'simple':
            rho = self.RK4_fast(rho, time_steps=2, tmax=0.5)

        if self.mu is None:
            mu_batch = None
        else:
            mu_batch = qutils.make_batch(tf.reshape(tf.cast(self.mu[self.timestep-1], dtype=tf.float32),(-1, 1, 1 )), self.batch_size)

        if self.dynamics_type is not 'autonomous':
            rho, sign, prob = ops.POVM(rho, theta=theta, phi=phi, Lambda=Lambda, deterministic=self.deterministic, mu=mu_batch)
            self.sign = sign 
        else:
            prob = qutils.make_batch(tf.reshape(tf.cast(1.0, dtype=tf.float32),(-1, 1, 1 )), self.batch_size)
            self.sign = qutils.make_batch(tf.reshape(tf.cast(1.0, dtype=tf.float32),(-1, 1, 1 )), self.batch_size)

        if self.timestep == 1:
            rho2=rho

        if self.verbosity == 'measure' or self.verbosity == 'all':
            print('Measure: ', -self.sign)
            print('Probability: ', prob)

        if self.verbosity == 'high' or self.verbosity == 'all':
            _, pauliZ, _ = self.pauli(rho2)
            print('Pauli Z = ',tf.reduce_mean(pauliZ).numpy())
            print('Fidelity = ',tf.reduce_mean(self.compute_fidelity(rho2)).numpy())

        if (self.verbosity == 'actions' or self.verbosity == 'all') and self.TEST == False:
            if self.timestep != 1:
                self.track_parameters.append(theta_VR[0,0,0])
            else:
                self.track_parameters.append(tf.cast(0.,tf.float32))
            self.track_parameters.append(phi1[0,0,0])
            self.track_parameters.append(phi2[0,0,0])
            self.track_parameters.append(phi3[0,0,0])
            self.track_parameters.append(phi4[0,0,0])
            self.track_parameters.append(theta1[0,0,0])
            self.track_parameters.append(theta2[0,0,0])
            self.track_parameters.append(theta3[0,0,0])
            self.track_parameters.append(theta4[0,0,0])
            self.track_parameters.append(beta1_r[0,0,0])
            self.track_parameters.append(beta2_r[0,0,0])
            self.track_parameters.append(beta3_r[0,0,0])
            self.track_parameters.append(beta1_i[0,0,0])
            self.track_parameters.append(beta2_i[0,0,0])
            self.track_parameters.append(beta3_i[0,0,0])

            self.track_meas.append(-self.sign[0])
            self.track_prob.append(prob[0])

            self.track_rho.append(rho2[0,:,:])

        return rho, rho2, self.sign, prob

    def stabilizer(self, state):
        z = tf.linalg.trace(tf.matmul(self.Sz, state))
        x = tf.linalg.trace(tf.matmul(self.Sx, state))
        s = 0.5 * (z + x)
        return tf.math.real(s), tf.math.real(z), tf.math.real(x)

    def compute_fidelity(self, rho):
        rho = qutils.ptrace(rho, axis=1, dim1=2, dim2=self.N, batch_size=self.batch_size) 
        prod = tf.linalg.matmul(rho, self.rho_target)
        f = tf.einsum('bii->b', prod)
        return tf.cast(f, tf.float32)

    def pauli(self, state):
        x = tf.linalg.trace(tf.matmul(self.PauliX, state))
        z = tf.linalg.trace(tf.matmul(self.PauliZ, state))
        p  = 0.5 * (x + z)
        return tf.math.real(p), tf.math.real(z), tf.math.real(x)

    def pauliy(self, state):
        y = tf.linalg.trace(tf.matmul(self.PauliY, state))
        return tf.math.real(y)


    def step(self, actions):
        self.timestep += 1
        done = (self.timestep == self.max_steps)
        self.actions = actions
        if self.protocol == 'sBs' or self.protocol == 'sBs_bias':        
            self.rho, self.rho2, self.sign, self.prob = self.QEC_sBs(actions=self.actions, rho=self.rho)
        elif self.protocol == 'ST':
            self.rho, self.rho2, self.sign, self.prob = self.QEC_ST(actions=self.actions, rho=self.rho)

        reward = 0.0
        R = 0 * self.R0
        try:
            if 'fidelity' in self.reward_type:
                if 'Pauli Z' in self.reward_type:
                    _, R, _ = self.pauli(self.rho2)
                    R += self.compute_fidelity(self.rho2)
                else:
                    R = self.compute_fidelity(self.rho2) 
            elif 'Pauli Z' in self.reward_type  and 'fidelity' not in self.reward_type:
                _, R, _ = self.pauli(self.rho2)
            elif 'g measures' in self.reward_type:
                R += -tf.cast(self.sign, tf.float32)
        except:
            R = 0 * self.R0
        reward += tf.reduce_mean(R).numpy()
        self.episode_rewards.append(reward)

        if self.evaluate:
            self.cumulated_reward = np.array(self.episode_rewards).sum()
            self.track_returns.append(self.cumulated_reward)
            self.track_returns.append(reward)
            self.track_progress(rho=self.rho2) 
            if (self.episode_count - 1) % self.save_interval == 0:
                self.save_progress()
                np.save("rewards", self.track_returns)
        else:
            if done:
                self.cumulated_reward = np.array(self.episode_rewards).sum()
                self.track_returns.append(self.cumulated_reward)

                if (self.episode_count - 1) % self.track_interval == 0:
                    self.track_progress(rho=self.rho2) 
                if (self.episode_count) % self.save_interval == 0:
                    self.save_progress()
                    np.save("returns", self.track_returns)
                if self.episode_count == 1 and self.verbosity == 'high':
                    print("Time taken per episode is approximately", time.time() - self.t0)
        info = {}
        info['prob'] = self.prob
        info['reward'] = R 
        self.observation = self.sign.numpy().flatten()
        return self.observation, reward, done, info

    def save_progress(self):
        np.save("pauliXZ.npy", self.track_pauli)
        np.save("pauliX.npy", self.track_pauliX)
        np.save("pauliZ.npy", self.track_pauliZ)
        np.save("pauliY.npy", self.track_pauliY)
        np.save("stabilizerXZ.npy", self.track_stabilizer)
        np.save("stabilizerX.npy", self.track_stabilizerX)
        np.save("stabilizerZ.npy", self.track_stabilizerZ)
        np.save("fidelity", self.track_fidelity)
        np.save("parameters.npy",self.track_parameters)
        np.save("measures.npy",self.track_meas)
        np.save("probabilities.npy",self.track_prob)
        np.save("density_matrix.npy",self.track_rho)

    def track_progress(self, rho=None):
        current_pauli, current_pauliZ, current_pauliX = self.pauli(rho)
        current_pauliY = self.pauliy(rho)
        current_pauli = tf.reduce_mean(current_pauli).numpy()
        current_pauliZ = tf.reduce_mean(current_pauliZ).numpy()
        current_pauliX = tf.reduce_mean(current_pauliX).numpy()
        current_pauliY = tf.reduce_mean(current_pauliY).numpy()
        self.track_pauli.append(current_pauli)
        self.track_pauliZ.append(current_pauliZ)
        self.track_pauliX.append(current_pauliX)
        self.track_pauliY.append(current_pauliY) 

        current_stabilizer, current_stabilizerZ, current_stabilizerX = self.stabilizer(rho)
        current_stabilizer = tf.reduce_mean(current_stabilizer).numpy()
        current_stabilizerZ = tf.reduce_mean(current_stabilizerZ).numpy()
        current_stabilizerX = tf.reduce_mean(current_stabilizerX).numpy()
        self.track_stabilizer.append(current_stabilizer)
        self.track_stabilizerZ.append(current_stabilizerZ)
        self.track_stabilizerX.append(current_stabilizerX)

        current_fidelity = self.compute_fidelity(rho) 
        current_fidelity = tf.reduce_mean(current_fidelity).numpy()
        self.track_fidelity.append(current_fidelity)
        print("#{} R={:.3f} F={:.3f} XZ={:.3f} X={:.3f} Z={:.3f} Sxz={:.3f} Sz={:3f} Sx={:3f}".format(self.episode_count,
                self.cumulated_reward, current_fidelity, current_pauli, current_pauliX, current_pauliZ,
                current_stabilizer, current_stabilizerZ, current_stabilizerX))

    def reset(self):
        self.episode_count +=1
        self.timestep = 0
        self.R = 0 * self.R0
        self.rho = self.rho0
        if isinstance(self.initial_state, float) or isinstance(self.initial_state, int):
            err = qutils.make_batch(tf.reshape(tf.constant(tf.cast(self.initial_state, dtype=tf.complex64), dtype=tf.complex64),
             (-1, 1, 1)), self.batch_size)
            displace_err = ops.displace(self.N, err, batch_size=self.batch_size, tensored=True)
            self.rho = qutils.matmul3(displace_err, self.rho ,qutils.dag(displace_err))
        self.observation = self.observation0
        self.episode_rewards = []
        self.sign = np.ones(shape=self.batch_size)
        return self.observation  
