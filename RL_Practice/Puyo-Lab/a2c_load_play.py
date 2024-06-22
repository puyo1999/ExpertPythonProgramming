## 학습된 신경망 파라미터를 가져와서 에이전트를 실행

import gym
import tensorflow as tf
from a2c_learn import A2Cagent

def main():
    env_name = "Pendulum-v0"
    env = gym.make(env_name)
    agent = A2Cagent(env)
    agent.load_weights('./save_weights/')

    time = 0
    state = env.reset()

    rewards = []
    while True:
        #env.render()

        action = agent.actor(tf.convert_to_tensor([state], dtype=tf.float32))[0][0]
        state, reward, done, _ = env.step(action)

        time += 1

        print(f"Time: {time}, Reward: {reward}\n")
        rewards.append(reward)

        if done:
            break

    env.close()

    agent.plot_rewards(rewards)

if __name__ == "__main__":
    main()