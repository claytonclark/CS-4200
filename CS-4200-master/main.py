import numpy as np
import matplotlib.pyplot as plt
import time

from env_pydash import PydashEnv
from a2c import ActorCritic
import tensorflow as tf

if __name__ == '__main__':

    env = PydashEnv()

    actor_critic = ActorCritic([env.observation_shape], env.action_shape)

    #actor_critic.actor = tf.keras.models.load_model('actor.h5',compile=True)
    #actor_critic.critic = tf.keras.models.load_model('critic.h5',compile=True)

    total_rewards = []

    for episode in range(1000):
        try:
            print(f'In Episode {episode}')

            actor_critic.train_episode(env, 10000)
            total_rewards.append(actor_critic.current_reward)

            print(np.mean(total_rewards), np.mean(total_rewards[-10:]), max(total_rewards))
            print(actor_critic.epsilon)
        except KeyboardInterrupt:
            print ('KeyboardInterrupt exception is caught')
            plt.plot(list(range(len(total_rewards))), total_rewards)
            plt.show()
            actor_critic.actor.save('actor.h5')
            time.sleep(5)
    actor_critic.actor.compile()
    actor_critic.critic.compile()
    actor_critic.actor.save('actor.h5')
    actor_critic.critic.save('critic.h5')
    
    
    ## Plot the average over time
    prev = 0
    mva = []

    k = 2 / (51)

    for el in total_rewards:

        if prev != 0:
            pres = prev * (1 - k) + el * k
        else:
            pres = el
        prev = pres
        mva.append(pres)
    plt.plot(list(range(len(total_rewards))), total_rewards, alpha=0.5, label="Episode Reward")
    plt.plot(list(range(len(mva))), mva, label="Moving Average")
    plt.legend()
    plt.show()