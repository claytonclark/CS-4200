import random
import numpy as np
import tensorflow as tf


class ActorCritic:

    def __init__(self, input_shape, num_actions, discount=0.99):

        initializer = tf.keras.initializers.GlorotUniform()
        # self.actor = tf.keras.models.Sequential([
        #     tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape, kernel_initializer=initializer),
        #     tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
        #     tf.keras.layers.Dense(num_actions, activation='softmax', kernel_initializer=initializer)])

        # self.critic = tf.keras.models.Sequential([
        #     tf.keras.layers.Dense(32, activation='relu', input_shape=input_shape, kernel_initializer=initializer),
        #     tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer),
        #     tf.keras.layers.Dense(1, activation='linear', kernel_initializer=initializer)])
        
        self.actor = self.create_actor(input_shape, num_actions)
        self.critic = self.create_critic(input_shape, 1)

        self.actor_optimizer = tf.keras.optimizers.Adam()
        self.critic_optimizer = tf.keras.optimizers.Adam()
        self.num_actions = num_actions
        self.discount = discount
        self.epsilon = 0.001
        #self.epsilon_decay = 0.9995
        self.epsilon_decay = 0.9975
        self.current_reward = 0

    def create_actor(self, state_shape, action_shape):
        learning_rate = 0.001
        init = tf.keras.initializers.HeUniform()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
        model.add(tf.keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
        model.add(tf.keras.layers.Dense(action_shape, activation='softmax', kernel_initializer=init))
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
        return model

    def create_critic(self, state_shape, output_shape):
        learning_rate = 0.001
        init = tf.keras.initializers.HeUniform()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(24, input_shape=state_shape, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
        model.add(tf.keras.layers.Dense(12, activation=tf.keras.layers.LeakyReLU(), kernel_initializer=init))
        model.add(tf.keras.layers.Dense(output_shape, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
        return model

    def forward(self, state):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        policy_dist = self.actor(state)
        value = self.critic(state)

        return tf.squeeze(policy_dist), tf.squeeze(value)

    def train_episode(self, env, max_steps):

        state = env.reset()
        rewards = []
        values = []
        log_probs = []

        with tf.GradientTape(persistent=True) as tape:

            for step in range(max_steps):

                policy_dist, value = self.forward(state)
                if np.random.random() > self.epsilon:
                    action = np.random.choice(self.num_actions, p=policy_dist.numpy())
                else:
                    action = env.sample_action()

                state, reward, done = env.step(action)

                log_prob = tf.math.log(policy_dist[action] + 1e-5)

                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)

                if done:
                    Qval = 0
                    break
                elif step == max_steps - 1:
                    _, Qval = self.forward(state)
                    Qval = Qval
                    break

            Qvals = np.zeros_like(values)
            for t in reversed(range(len(values))):
                Qval = rewards[t] + self.discount * Qval
                Qvals[t] = Qval

            Qvals = (Qvals - np.mean(Qvals)) / (np.std(Qvals) + np.finfo(
                np.float32).eps.item())  # np.finfo(np.float32).eps.item() is small epsilon value

            advantages = []
            for Qval, value in zip(Qvals, values):
                advantages.append(Qval - value)

            actor_loss = 0
            for log_prob, advantage in zip(log_probs, advantages):
                actor_loss -= (log_prob * advantage) / len(log_probs)

            critic_loss = 0.5 * tf.math.reduce_mean(tf.math.pow(advantages, 2))

            ac_loss = actor_loss + critic_loss

        actor_grads, critic_grads = tape.gradient(ac_loss,
                                                  [self.actor.trainable_variables, self.critic.trainable_variables])

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        self.epsilon *= self.epsilon_decay

        self.current_reward = sum(rewards)