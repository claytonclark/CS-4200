import numpy as np

from env_pydash import PydashEnv
from a2c import ActorCritic
import tensorflow as tf

if __name__ == '__main__':

    env = PydashEnv()

    actor_critic = ActorCritic([env.observation_shape], env.action_shape)

    #actor_critic.actor = tf.keras.models.load_model('models/actor.h5')
    #actor_critic.critic = tf.keras.models.load_model('models/critic.h5')

    total_rewards = []

    for episode in range(1800):

        print(f'In Episode {episode}')

        actor_critic.train_episode(env, 1000)
        total_rewards.append(actor_critic.current_reward)

        print(np.mean(total_rewards), np.mean(total_rewards[-10:]), max(total_rewards))
        print(actor_critic.epsilon)

    actor_critic.actor.save('actor.h5')