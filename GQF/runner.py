import tensorflow as tf
import numpy as np

from feedback_GRAPE import FeedbackGRAPE

bias_initializer = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01)

feedback_grape = FeedbackGRAPE(N=100, Delta=0.34,
    max_steps=21,
    batch_size=8,
    num_controls=15,
    num_training_episodes=101,
    num_evaluation_episodes=1,
    lr=1E-4,
    protocol='sBs',
    deterministic=False,
    mu=None,
    reward_type = 'final fidelity',
    dynamics_type='Sivak2023',
    Hamiltonian='off',
    initialized_values=True,
    model = 'RNN',
    memory_length = 1,
    verbosity='low',
    saving = 'periodic and best',
    noise_level = 'high',
    TEST=False
   )

feedback_grape.run(task='train', retrain=False)

