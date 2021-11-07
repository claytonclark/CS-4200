import tensorflow as tf
import numpy as np
import os
from tensorflow import keras

env_action_space = 1
#env_observation_space = 

# An episode a full game
train_episodes = 300

def create_actor(state_shape, action_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(action_shape, activation='softmax', kernel_initializer=init))
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def create_critic(state_shape, output_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(24, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
    model.add(keras.layers.Dense(output_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model

def one_hot_encode_action(action, n_actions):
    encoded = np.zeros(n_actions, np.float32)
    encoded[action] = 1
    return encoded

def main():
    actor_checkpoint_path = "training_actor/actor_cp.ckpt"
    critic_checkpoint_path = "training_critic/critic_cp.ckpt"