"""
MultiWorkerActorCritic.py
"""

import numpy as np
import tensorflow as tf
import gym
import tensorflow_probability as tfp
from multiprocessing import Process, Queue, Barrier, Lock
import keras.losses as kls

env = gym.make('CartPole-v0')
low = env.observation_space.low
high = env.observation_space.high

class critic(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.v = tf.keras.layers.Dense(1, activation=None)

    def call(self, input_data):
        x = self.d1(input_data)
        v = self.v(x)
        return v

class actor(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        # self.d2 = tf.keras.layers.Dense(32,activation='relu')
        self.a = tf.keras.layers.Dense(2, activation='softmax')

    def call(self, input_data):
        x = self.d1(input_data)
        # x = self.d2(x)
        a = self.a(x)
        return a

class agent():
    def __init__(self, gamma = 0.99):
        self.gamma = gamma
        self.a_opt = tf.keras.optimizers.RMSprop(learning_rate=7e-3)
        self.c_opt = tf.keras.optimizers.RMSprop(learning_rate=7e-3)
        self.actor = actor()
        self.critic = critic()

    def act(self, state):
        prob = self.actor(np.array([state]))
        prob = prob.numpy()
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss(self, probs, actions, td):

        probability = []
        log_probability = []
        for pb, a in zip(probs, actions):
            dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
            log_prob = dist.log_prob(a)
            prob = dist.prob(a)
            probability.append(prob)
            log_probability.append(log_prob)

        # print(probability)
        # print(log_probability)

        p_loss = []
        e_loss = []
        td = td.numpy()
        # print(td)
        for pb, t, lpb in zip(probability, td, log_probability):
            t = tf.constant(t)
            policy_loss = tf.math.multiply(lpb, t)
            entropy_loss = tf.math.negative(tf.math.multiply(pb, lpb))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)
        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)
        # print(p_loss)
        # print(e_loss)
        loss = -p_loss - 0.0001 * e_loss
        # print(loss)
        return loss

    def learn(self, states, actions, discnt_rewards):
        discnt_rewards = tf.reshape(discnt_rewards, (len(discnt_rewards),))

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            p = self.actor(states, training=True)
            v = self.critic(states, training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discnt_rewards, v)
            # print(discnt_rewards)
            # print(v)
            # print(td.numpy())
            a_loss = self.actor_loss(p, actions, td)
            c_loss = 0.5 * kls.mean_squared_error(discnt_rewards, v)
        grads1 = tape1.gradient(a_loss, self.actor.trainable_variables)
        grads2 = tape2.gradient(c_loss, self.critic.trainable_variables)
        self.a_opt.apply_gradients(zip(grads1, self.actor.trainable_variables))
        self.c_opt.apply_gradients(zip(grads2, self.critic.trainable_variables))
        return a_loss, c_loss

def preprocess1(states, actions, rewards, gamma, s_queue, a_queue, r_queue, lock):
    discnt_rewards = []
    sum_reward = 0
    rewards.reverse()
    for r in rewards:
      sum_reward = r + gamma*sum_reward
      discnt_rewards.append(sum_reward)
    discnt_rewards.reverse()
    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.int32)
    discnt_rewards = np.array(discnt_rewards, dtype=np.float32)
    #exp = np.array([states, actions,discnt_rewards])
    lock.acquire()
    s_queue.put(states)
    a_queue.put(actions)
    r_queue.put(discnt_rewards)
    lock.release()

def preprocess2(s_queue, a_queue, r_queue):
    states = []
    while not s_queue.empty():
        states.append(s_queue.get())

    actions = []
    while not a_queue.empty():
        actions.append(a_queue.get())
    dis_rewards = []
    while not r_queue.empty():
        dis_rewards.append(r_queue.get())

    state_batch = np.concatenate(*(states,), axis=0)
    action_batch = np.concatenate(*(actions,), axis=None)
    reward_batch = np.concatenate(*(dis_rewards,), axis=None)
    # exp = np.transpose(exp)

    return state_batch, action_batch, reward_batch

def runner(barrier, lock, s_queue, a_queue, r_queue):
    tf.random.set_seed(360)
    agentoo7 = agent()
    steps = 2000
    ep_reward = []
    total_avgr = []

    for s in range(steps):
        done = False
        state = env.reset()
        total_reward = 0
        all_aloss = []
        all_closs = []
        rewards = []
        states = []
        actions = []

        while not done:
            action = agentoo7.act()
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)
            states.append(state)
            actions.append(action)

            state = next_state
            total_reward += reward

            if done:
                ep_reward.append(total_reward)
                avg_reward = np.mean(ep_reward[-100])
                total_avgr.append(avg_reward)
                print('total reward after {} step is {} and avg reward is {}'.format(s, total_reward, avg_reward))

                preprocess1(states, actions, rewards, 1, s_queue, a_queue, r_queue, lock)
                b = barrier.wait()
                if b == 0:
                    if (s_queue.qsize() == 10) & (a_queue.qsize() == 10) & (r_queue.qsize() == 10):
                        print(s_queue.qsize())
                        print(a_queue.qsize())
                        print(r_queue.qsize())
                        state_batch, action_batch, reward_batch = preprocess2(s_queue, a_queue, r_queue)
                        # print(state_batch)
                        # print(action_batch)
                        # print(reward_batch)
                        al, cl = agentoo7.learn(state_batch, action_batch, reward_batch)
                        all_aloss.append(al)
                        all_closs.append(cl)
                        print(f"al{al}")
                        print(f"cl{cl}")

                barrier.wait()


barrier = Barrier(10)
s_queue = Queue()
a_queue = Queue()
r_queue = Queue()
lock = Lock()

processes = []
for i in range(10):
    worker = Process(target=runner, args=(barrier, lock, s_queue, a_queue, r_queue))
    processes.append(worker)
    worker.start()

for process in processes:
    process.join()