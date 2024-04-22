import tensorflow as tf
import numpy as np

from GKP_environment import GKPEnv

class FeedbackGRAPE:
    def __init__(self, 
                 N=100, 
                 Delta=0.2, 
                 max_steps=21, 
                 batch_size=5, 
                 num_controls=15, 
                 num_training_episodes=500, 
                 num_evaluation_episodes=10, 
                 lr=1E-4,
                 initialized_values=True,
                 dynamics=True,
                 deterministic=False,
                 mu=None,
                 protocol='sBs',
                 dynamics_type='Sivak2023',
                 Hamiltonian='off',
                 chi=None,
                 K_s=None,                 
                 reward_type='final fidelity',
                 initial_state='zero logical',
                 model=None,
                 memory_length = 1,
                 verbosity='low',
                 saving='best',
                 noise_level = 'low',
                 TEST=False):

        '''
        Class implementing the Feedback-GRAPE algorithm according to PRX Quantum 4, 030305 (2023).

        Args:
            N (int, optional): Number of Fock states for the bosonic system describing the cavity. Default: 100
            
            Delta (float, optional): Parameter for the physical GKP code. Default: 0.2
            
            max_steps (int, optional): Number of half-cycles (and measurements) per trajectory, included the first step. Default: 21
            
            batch_size (int, optional): Batch-size of the vectorized environment. Default: 5
            
            num_controls (int): Number of control parameters. Default: 15
            
            num_training_episodes (int): Number of training episodes. Default: 500
            
            num_evaluation_episodes (int, optional): Number of evaluation episodes. Default: 10
            
            lr (float, optional): Learning rate. Default: 1E-4
            
            initialized_values (bool, optional): Initial values for the gates' parameters. Default: True
            
            dynamics (bool, optional): Dissipative dynamics. Default: True
            
            deterministic (bool, optional): Deterministic measurements' outcomes. If True it post-selects trajectories according to 
                the values given in the argument 'mu'. Default: False
            
            mu (array(float), optional): Array of measurements' outcomes in case of deterministic dynamics.
            
            protocol (str, optional): Quantum error correction protocol to be used. Currently supports only 'sBs' protocol
            
            dynamics_type (str, optional): Dynamics type to be used in the simulation. Possible values are 'Sivak2023', 'Ibarcq2020', 
                'autonomous', 'simple', 'longer'. Default: 'Sivak2023'
            
            Hamiltonian (str, optional):  If we want to add the time evolution with the Hamiltonian in the noisy dynamics. Default: 'off'
            
            chi (float, optional): Value of dispersive coupling term in the Hamiltonian in case it is used. Default: None
            
            K_s (float, optional): Value of the Kerr constant in case used the Hamiltonian dynamics. Default: None
            
            reward_type (str, optional): Type of reward for the neural network training. Default: 'final fidelity'
            
            initial_state (str, list(complex), optional): Initial state for the simulation. Possible values are: 'zero logical', 'one logical', 'x logical', '-x logical', 'y logical', '-y logical'. Alternatively it is possible to give a list of two complex numbers (a,b) such that the initial state is |i > = a |0_L > + b | 1_L >, with a^2 + b^2 = 1. Default: 'zero logical'
            
            model (str, Model): the model used for training. Possible values: 'rNN', 'ffNN', or a custom keras model.  
            
            memory_length (int, optional): Number of measurements to feed as input in the NN at each step. Default: 1
            
            verbosity (str, optional): Verbosity of the output generated. Possible values: 'low', 'high', 'all', 'measurements'. Default: 'low'
            
            saving (str, optional):  . Default: 'best'
            
            noise_level (str, optional): Pre-defined noise levels in the dynamics. Possible values: 'low', 'medium', 'high'. Default: 'low'
            
            TEST (bool, optional): Performing an evaluation test with constant parameters. Default: False
        '''

        self.N = N
        self.Delta = Delta
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.initialized_values = initialized_values
        self.dynamics = dynamics
        self.Hamiltonian = Hamiltonian
        self.chi = chi
        self.K_s = K_s
        self.deterministic = deterministic
        self.mu = mu
        self.protocol = protocol
        self.num_controls = num_controls
        self.num_training_episodes = num_training_episodes
        self.num_evaluation_episodes = num_evaluation_episodes
        self.previous_reward = 0.0 
        self.lr = lr
        self.dynamics_type = dynamics_type
        self.reward_type = reward_type
        self.model = model 
        self.memory_length = memory_length
        self.verbosity = verbosity
        self.saving = saving
        self.initial_state = initial_state
        self.noise_level = noise_level
        self.TEST = TEST

    def play_an_episode(self, model, env):
        obs_input = np.reshape(env.reset(), (self.batch_size, 1, self.memory_length))
        log_probs = R = tf.zeros((self.batch_size, 1, 1), dtype=tf.float32)
        obs_ok = tf.Variable(tf.zeros((self.batch_size,self.memory_length), dtype=tf.complex64))
        done = False
        ii = 0
        while not done:
            action = tf.cast(model(obs_input), dtype=tf.complex64)
            obs_output, reward, done, info = env.step(action)
            obs_input = obs_output
            if self.memory_length > 1:
                obs_ok[:,0:self.memory_length-2].assign(obs_ok[:,1:self.memory_length-1])
                obs_ok[:,self.memory_length-1].assign(obs_output) 
                obs_input = obs_ok
            if self.verbosity == 'high':
                print(f"Actions: {action}")
                print('measures:',obs_output)
                print('Probabilities:',tf.reshape(info['prob'], self.batch_size))
            obs_input = np.reshape(obs_input, (self.batch_size, 1, self.memory_length))
            prob = tf.reshape(info['prob'], self.batch_size)
            rewards = info['reward']
            log_probs += tf.math.log(prob)
            R += rewards
            ii += 1
        return log_probs, R, rewards
    

    def save_best_agent(self, model, reward, previous_reward):
        if reward > previous_reward:
            model.save("best_model.h5") 
            if self.verbosity == 'high':
                print("New best agent - reward = ", reward)


    def train(self, retrain=False, model=None):
        env = GKPEnv(N=self.N, 
                     Delta=self.Delta, 
                     noise_level=self.noise_level, 
                     max_steps=self.max_steps, 
                     initialized_values = self.initialized_values,
                     memory_length=self.memory_length, 
                     protocol = self.protocol,
                     batch_size=self.batch_size,
                     dynamics=self.dynamics,
                     deterministic=self.deterministic,
                     mu=self.mu,
                     dynamics_type=self.dynamics_type,
                     Hamiltonian=self.Hamiltonian,
                     chi=self.chi,
                     K_s=self.K_s,
                     track_interval=1, 
                     save_interval=10, 
                     reward_type=self.reward_type, 
                     evaluate=False, 
                     initial_state=self.initial_state, 
                     verbosity=self.verbosity)

        bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)
        if retrain:
                model = tf.keras.models.load_model(model)
        else:
            if self.model == None:
                raise(ValueError)
            elif self.model == 'rNN':
                model = tf.keras.Sequential([
                tf.keras.layers.GRU(30, batch_input_shape=(self.batch_size, 1, 1), stateful=True, activation='tanh'),
                tf.keras.layers.Dense(30, activation='tanh'),
                tf.keras.layers.Dense(30, activation='tanh'),
                tf.keras.layers.Dense(self.num_controls, bias_initializer=bias_initializer, activation='tanh')])
            elif self.model == 'ffNN':
                model = tf.keras.Sequential([
                tf.keras.layers.Input(batch_input_shape=(self.batch_size,1,self.memory_length)),
                tf.keras.layers.Dense(30, activation='tanh'),
                tf.keras.layers.Dense(30, activation='tanh'),
                tf.keras.layers.Dense(self.num_controls, bias_initializer=bias_initializer, activation='tanh')])
            else:
                model = self.model

        for episode in range(self.num_training_episodes):
            optimizer = tf.optimizers.Adam(learning_rate=self.lr)
            if self.model != 'ffNN':
                model.reset_states()
                            
            with tf.GradientTape(persistent=False) as tape:
                log_probs, reward, final_reward = self.play_an_episode(model, env)
                if 'final' in self.reward_type:
                    R = final_reward 
                elif 'cumulative' in self.reward_type:
                    R = reward
                else:
                    raise(NotImplementedError)
                loss = tf.reduce_mean(-R) + tf.reduce_mean(log_probs * tf.stop_gradient(-R))
            y = model.trainable_variables
            grads = tape.gradient(loss, y)
            optimizer.apply_gradients(zip(grads, y))

            print(f"Episode: {episode} Loss: {loss} Reward: {tf.reduce_mean(reward).numpy()} "
                  f"Final reward: {tf.reduce_mean(final_reward).numpy()}")
            if self.saving == 'periodic':
                if episode % 10 == 0:
                    model.save("model.h5")
            elif self.saving == 'periodic and best':
                if episode % 10 == 0:
                    model.save("model.h5")
            rew = tf.reduce_mean(final_reward)
            if rew > self.previous_reward:
                if self.saving == 'best':
                    self.save_best_agent(model, rew, self.previous_reward)
                elif self.saving == 'last':
                    if episode == self.num_training_episodes-1 :
                        model.save('last_model.h5')
                elif self.saving == 'best and last':
                    self.save_best_agent(model, rew, self.previous_reward)
                    model.save('last_model.h5')
                elif self.saving == 'periodic and best':
                    self.save_best_agent(model, rew, self.previous_reward)


    def evaluate(self, model):
        env = GKPEnv(N=self.N, 
                     Delta=self.Delta, 
                     noise_level=self.noise_level, 
                     max_steps=self.max_steps, 
                     initialized_values = self.initialized_values,
                     memory_length=self.memory_length,
                     protocol=self.protocol,
                     batch_size=self.batch_size,
                     dynamics=self.dynamics,
                     deterministic=self.deterministic,
                     mu=self.mu,
                     dynamics_type=self.dynamics_type,
                     Hamiltonian=self.Hamiltonian,
                     chi=self.chi,
                     K_s=self.K_s,
                     track_interval=1, 
                     save_interval=10, 
                     reward_type=self.reward_type, 
                     initial_state=self.initial_state, 
                     verbosity=self.verbosity, 
                     evaluate=True,
                     TEST=self.TEST)

        if self.TEST:
            for episode in range(self.num_evaluation_episodes):
                obs = env.reset()
                obs_ok = tf.zeros(self.batch_size,self.memory_length)
                done = False
                ii = 0
                while not done:
                    obs = np.reshape(obs, (self.batch_size, 1, self.memory_length))
                    obs, reward, done, info = env.step(actions=None)
                    if self.memory_length > 1:
                        obs_ok[:,0:self.memory_length-2] = obs_ok[:,1:self.memory_length-1]
                        obs_ok[:,self.memory_length-1] = obs
                        obs = obs_ok
                    ii += 1
        else:    
            model = tf.keras.models.load_model(model)
            for episode in range(self.num_evaluation_episodes):
                model.reset_states()
                log_probs, reward, final_reward = self.play_an_episode(model, env)
            

    def run(self, task, retrain=False, model='model.zip'):
        if task == 'train':
            self.train(retrain, model=model)
        elif task == 'evaluate':
            if self.TEST:
                self.evaluate(model=None)
            else:
                self.evaluate(model)
        else:
            print("Choose either train or evaluate")




