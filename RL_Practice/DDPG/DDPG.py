import random
from collections import deque
from math import sqrt

import gym
from gym import spaces
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, concatenate
from keras.initializers import RandomUniform

import matplotlib.pyplot as plt

output_init = RandomUniform(-3*10e-3, 3+10e-3)

class OUNoise:
    def __init__(self, act_dim, mu=0, theta=0.15, sigma=0.2):
        self.act_dim = act_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.act_dim) * self.mu
        self.reset()
    def reset(self):
        self.state = np.ones(self.act_dim) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    def __init__(self, buf_size):
        self.buf_size = int(buf_size)
        self.buffer = deque(maxlen=self.buf_size)
    def sample(self, batch_size):
        size = batch_size if len(self.buffer) > batch_size else len(self.buffer)
        return random.sample(self.buffer, size)
    def clear(self):
        self.buffer.clear()
    def append(self, transition):
        self.buffer.append(transition)
    def __len__(self):
        return len(self.buffer)


class Actor(keras.Model):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        act_dim = np.squeeze(act_dim)
        self.dense1 = Dense(128, activation='relu', kernel_initializer=RandomUniform(-1/sqrt(128), 1/sqrt(128)))
        self.dense2 = Dense(128, activation='relu', kernel_initializer=RandomUniform(-1/sqrt(128), 1/sqrt(128)))
        self.output_layer = Dense(act_dim, activation='linear', kernel_initializer=output_init)
        self.build(input_shape=(None,) + obs_dim)
        self.summary()

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)

class Critic(keras.Model):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        self.o_dense1 = Dense(128, activation='relu', kernel_initializer=RandomUniform(-1/sqrt(128), 1/sqrt(128)))
        self.o_dense2 = Dense(128, activation='relu', kernel_initializer=RandomUniform(-1/sqrt(128), 1/sqrt(128)))
        self.output_layer = Dense(1, activation='linear', kernel_initializer=output_init)
        self.build(input_shape=[(None,) + obs_dim, (None,)+act_dim])
        self.summary()
    def call(self, inputs, training=None, mask=None):
        obs, action = inputs
        z = tf.concat([obs, action], axis=1)
        x = self.o_dense1(z)
        x = self.o_dense2(z)
        return self.output_layer(x)

class Agent:
    def __init__(self, env, obs_dim, act_dim, steps, gamma=0.99, buf_size=1e6, batch_size=64, tau=0.001):
        self.env = env
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.steps = steps
        self.gamma = gamma
        self.buf_size = buf_size
        self.batch_size = batch_size
        self.tau = tau

        self.noise = OUNoise(act_dim)
        self.replay_buffer = ReplayBuffer(buf_size=buf_size)

        self.actor = Actor(obs_dim, act_dim)
        self.actor_target = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)

        #self.soft_target_update(tau=1)
        #self.soft_target_update(tau=0.01)

        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.actor_optimizer = tf.keras.optimizers.Adam(lr=0.0001)
        self.critic_optimizer = tf.keras.optimizers.Adam(lr=0.001)



    def train(self):
        global_steps = 0
        epochs = 0
        rewards_list = []
        while global_steps < self.steps:
            ob = self.env.reset()
            rewards = 0
            epoch_step = 0
            while True:
                ac = self.get_action(ob)
                next_ob, reward, done, _ = self.env.step(ac)

                transitions = (ob, ac, next_ob, reward, done)
                self.replay_buffer.append(transitions)

                ob = next_ob
                rewards += reward
                global_steps += 1
                epoch_step += 1

                if global_steps >= 4000:
                    if global_steps == 4000:
                        print("train start")
                    transitions = self.replay_buffer.sample(batch_size=self.batch_size)
                    #self.learn(*map(lambda x: np.vstack(x).astype('float32'), np.transpose(transitions)))
                    self.learn(ob, ac, next_ob, reward, done=False)
                    self.soft_target_update()

                if done:
                    rewards_list.append(rewards/epoch_step)
                    print(f'{epochs} epochs, avg reward is {rewards/epoch_step}')
                    epochs += 1
                    break

        plt.plot(rewards_list, "r")
        plt.xlabel('epochs')
        plt.ylabel('avg reward (total_reward / steps')
        plt.title('deep learning phase')
        plt.savepig("ddpg_learning_pendulum-v0.png")
        self.actor.save_weights("./ddpg_actor/actor", overwrite=True)
        plt.close()

    def test(self, epochs=50):
        global_steps = 0
        epoch = 0
        rewards_list = []
        while epoch < epochs:
            ob = self.env.reset()
            rewards = 0
            epoch_step = 0
            while True:
                ac = self.get_action(ob, train_mode=False)
                next_ob, reward, done, _ = self.env.step(ac)

                transitions = (ob, ac, next_ob, reward, done)
                self.replay_buffer.append(transitions)

                ob = next_ob
                rewards += reward
                global_steps += 1
                epoch_step += 1

                # env.render()

                if done:
                    rewards_list.append(rewards/epoch_step)
                    print(f"# {epoch} epochs avg reward is {rewards/epoch_step}")
                    epoch += 1
                    break

    @tf.function
    def learn(self, ob, ac, next_ob, reward, done):
        next_ac = tf.clip_by_value(self.actor_target(next_ob), self.env.action_space.low, self.env.action_space.high)
        q_target = self.critic_target([next_ob, next_ac])
        y = reward + (1-done) * self.gamma * q_target
        with tf.GradientTape() as tape_c:   # for train critic
            q = self.critic([ob, ac])
            q_loss = self.loss_fn(y, q)
        grads_c = tape_c.gradient(q_loss, self.critic.trainable_weights)

        with tf.GradientTape() as tape_a:   # for train actor
            a = self.actor(ob)
            q_for_grad = -tf.reduce_mean(self.critic([ob,a]))
        grads_a = tape_a.gradient(q_for_grad, self.actor.trainable_weights)


        self.critic_optimizer.apply_gradients(zip(grads_c, self.critic.trainable_weights))
        self.actor_optimizer.apply_gradients(zip(grads_a, self.actor.trainable_weights))


    def get_action(self, ob, train_mode=True):
        if train_mode:
            return np.clip(self.actor(ob[np.newaxis])[0] + self.noise.noise(),
                           self.env.action_space.low,
                           self.env.action_space.high)
        else:
            return np.clip(self.actor(ob[np.newaxis])[0] + self.noise.noise(),
                           self.env.action_space.low,
                           self.env.action_space.high)

    def soft_target_update(self, tau=None):
        tau = self.tau if tau is None else tau
        #actor_tmp = tau * np.array(self.actor.get_weights()) + (1.- tau) * np.array(self.actor_target.get_weights())
        actor_tmp = tau * np.array(self.actor.get_weights()) + (1. - tau) * np.array(self.actor_target.get_weights())
        critic_tmp = tau * np.array(self.critic.get_weights()) + (1. - tau) * np.array(self.critic_target.get_weights())

        self.actor_target.set_weights(actor_tmp)
        self.critic_target.set_weithgs(critic_tmp)

if __name__ == '__main__':
    env = gym.make('Pendulum-v0')

    obs_dim = env.observation_space.shape
    act_dim = None

    if isinstance(env.action_space, spaces.Box):
        act_type = 'continuous'
        act_dim = env.action_space.shape
    elif isinstance(env.action_space, spaces.Discrete):
        act_type = 'discrete'
        act_dim = env.action_space.n
    else:
        raise NotImplementedError('Not implemented GG')

    agent = Agent(env, obs_dim, act_dim, 100000)
    agent.train()
    agent.test()
