import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow as tf
import keras
import keras as K
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
import gym
import numpy as np
import random as rand
LOSS_CLIPPING = 0.1
class Memory(object):
    """class Memory:

    """
    def __init__(self):
        self.batch_s = []
        self.batch_a = []
        self.batch_r = []
        self.batch_gae_r = []
        self.batch_s_ = []
        self.batch_done = []
        self.GAE_CALCULATED_Q = False

    def get_batch(self, batch_size):
        for _ in range(batch_size):
            s,a,r,gae_r,s_,d = [],[],[],[],[],[]
            pos = np.random.randint(len(self.batch_s))
            s.append(self.batch_s[pos])
            a.append(self.batch_a[pos])
            gae_r.append(self.batch_gae_r[pos])
            s_.append(self.batch_s_[pos])
            d.append(self.batch_done[pos])
        return s,a,r,gae_r,s_,d

    def store(self, s, a, s_, r, done):
        """
        :param s:
        :param a:
        :param s_:
        :param r:
        :param done:
        :return:
        """
        self.batch_s.append(s)
        self.batch_a.append(a)
        self.batch_r.append(r)
        self.batch_s_.append(s_)
        self.batch_done.append(done)

    def clear(self):
        self.batch_s.clear()
        self.batch_a.clear()
        self.batch_r.clear()
        self.batch_s_.clear()
        self.batch_done.clear()
        self.GAE_CALCULATED_Q = False

    @property
    def cnt_samples(self):
        return len(self.batch_s)

class Agent(object):
    def __init__(self, action_n, state_dim, training_batch_size):
        #self.env = gym.make('CartPole-v1')
        self.action_n = action_n
        self.state_dim = state_dim
        self.value_size = 1

        self.node_num = 32
        self.learning_rate_actor = 0.0005
        self.learning_rate_critic = 0.0005
        self.epochs_cnt = 5

        self.discount_rate = 0.98
        self.smooth_rate = 0.95
        self.episode_num = 500

        #CONSTANTS
        self.TRAINING_BATCH_SIZE = training_batch_size
        self.TARGET_UPDATE_ALPHA = 0.95
        self.GAMMA = 0.99
        self.GAE_LAMBDA = 0.95
        self.CLIPPING_LOSS_RATIO = 0.1
        self.ENTROPY_LOSS_RATIO = 0.001
        self.TARGET_UPDATE_ALPHA = 0.9

        self.model_actor = self.build_model_actor()
        self.model_critic = self.build_model_critic()

        self.model_actor_old = self.build_model_actor()
        self.model_actor_old.set_weights(self.model_actor.get_weights())

        self.dummy_advantage = np.zeros((1,1))
        self.dummy_old_prediction = np.zeros((1, self.action_n))

        self.memory = Memory()


    '''class MyModel(keras.Model):
        def train_step(self, data):
            in_datas, out_action_probs = data
            states, action_matrixs, advantages = in_datas[0], in_datas[1], in_datas[2]
            with tf.GradientTape() as tape:
                y_pred = self(states, training=True)
                new_policy = K.max(action_matrixs*y_pred, axis=-1)
                old_policy = K.max(action_matrixs*out_action_probs, axis=-1)
                r = new_policy/(old_policy)
                clipped = K.clip(r, 1-LOSS_CLIPPING, 1+LOSS_CLIPPING)
                loss = -K.minimum(r*advantages, clipped*advantages)
            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    '''

    def build_model_actor(self):

        state = Input(shape=self.state_dim, name='state')
        advantage = Input(shape=(1,), name='advantage_input')
        old_prediction = K.layers.Input(shape=(self.action_n,), name= 'old_prediction_input')

        dense = Dense(self.node_num, activation='relu', name='dense1')(state)
        dense = Dense(self.node_num, activation='relu', name='dense2')(dense)
        policy = Dense(self.action_n, activation='softmax', name='actor_output_layer')(dense)

        model = K.Model(inputs=[state, advantage, old_prediction], outputs=policy)
        model.compile(
            optimizer='Adam',
            loss = self.ppo_loss(advantage, old_prediction)
        )
        model.summary()
        return model

    def build_model_critic(self):
        state = Input(shape=(self.state_dim), name='state_input')
        dense = Dense(32, activation='relu')(state)
        dense = Dense(32, activation='relu')(dense)

        V = K.layers.Dense(1, name='actor_output_layer')(dense)

        model = K.Model(inputs=[state], outputs=V)
        model.compile(
            optimizer='Adam',
            loss = 'mean_squared_error'
        )
        model.summary()
        return model

    def ppo_loss(self, advantage, old_pred):
        """
        :param advantage:
        :param old_prediction:
        :return:
        """
        def loss(y_true, y_pred):
            prob = y_true * y_pred
            old_prob = y_true * old_pred
            ratio = prob / (old_prob + 1e-10)
            clip_ratio = K.backend.clip(ratio, min_value=1-self.CLIPPING_LOSS_RATIO, max_value=1+self.CLIPPING_LOSS_RATIO)
            surrogate1 = ratio * advantage
            surrogate2 = clip_ratio * advantage
            entropy_loss = (prob * K.backend.log(prob+1e-10))
            ppo_loss = -K.backend.mean(K.backend.minimum(surrogate1, surrogate2)+self.ENTROPY_LOSS_RATIO * entropy_loss)
            return ppo_loss
        return loss

    def make_gae(self):
        gae = 0
        mask = 0
        for i in reversed(range(self.memory.cnt_samples)):
            mask = 0 if self.memory.batch_done[i] else 1 # mask = 1-done
            v = self.get_v(self.memory.batch_s[i])
            delta = self.memory.batch_r[i] + self.GAMMA * self.get_v(self.memory.batch_s_[i]) * mask - v
            gae = delta + self.GAMMA * self.GAE_LAMBDA * mask * gae
            self.memory.batch_gae_r.append(gae + v)
        self.memory.batch_gae_r.reverse()
        self.memory.GAE_CALCULATED_Q = True

    def update_target_network(self):
        alpha = self.TARGET_UPDATE_ALPHA
        actor_weights = np.array(self.model_actor.get_weights(), dtype=object)
        actor_target_weights = np.array(self.model_actor_old.get_weights(), dtype=object)
        new_weights = alpha*actor_weights + (1-alpha)*actor_target_weights
        self.model_actor_old.set_weights(new_weights)

    def choose_action(self, state):
        """chooses an action within the action space given a state.
        The action is chosen by random with the weightings accoring to the probability
        params:
            :state: np.array of the states with state_dim length
        """
        assert isinstance(state, np.ndarray)
        # reshape for predict_on_batch which requires 2d-arrays
        state = np.reshape(state, [-1, self.state_dim[0]])
        # the probability list for each action is the output of the actor network given a state
        prob = self.model_actor.predict_on_batch([state, self.dummy_advantage, self.dummy_old_prediction]).flatten()
        # action is chosen by random with the weightings accoring to the probability
        action = np.random.choice(self.action_n, p=prob)
        return action

    def store_transition(self, s, a, s_, r, done):
        self.memory.store(s, a, s_, r, done)

    def get_v(self,state):
        """Returns the value of the state.
        Basically, just a forward pass though the critic networtk
        """
        s = np.reshape(state,(-1, self.state_dim[0]))
        v = self.model_critic.predict_on_batch(s)
        return v

    def get_old_prediction(self, state):
        """Makes an old prediction (an action) given a state on the actor_old_network.
        This is for the train_network --> ppo_loss
        """
        state = np.reshape(state, (-1, self.state_dim[0]))
        return self.model_actor_old.predict_on_batch([state, self.dummy_advantage, self.dummy_old_prediction])


    def train_network(self):
        """
        1. Get GAE rewards
        2. reshape batches s,a,gae_r baches
        3. get value of state
        4. calc advantage
        5. get "old" predicition (of target network)
        6. fit actor and critic network
        7. soft update target "old" network
        :return:
        """
        if not self.memory.GAE_CALCULATED_Q:
            self.make_gae()

        states,actions,rewards,gae_r,next_states,dones = self.memory.get_batch(self.TRAINING_BATCH_SIZE)

        # create np array batches for training
        batch_s = np.vstack(states)
        batch_a = np.vstack(actions)
        batch_gae_r = np.vstack(gae_r)
        # get values of states in batch
        batch_v = self.get_v(batch_s)
        # calc advantages. required for actor loss.
        batch_advantage = batch_gae_r - batch_v
        batch_advantage = K.utils.normalize(batch_advantage)  #
        # calc old_prediction. Required for actor loss.
        batch_old_prediction = self.get_old_prediction(batch_s)
        # one-hot the actions. Actions will be the target for actor.
        batch_a_final = np.zeros(shape=(len(batch_a), self.action_n))
        batch_a_final[:, batch_a.flatten()] = 1

        #commit training
        self.model_actor.fit(x=[batch_s, batch_advantage, batch_old_prediction], y=batch_a_final, verbose=0)
        self.model_critic.fit(x=batch_s, y=batch_gae_r, epochs=1, verbose=0)
        #soft update the target network(aka actor_old).
        self.update_target_network()

#%%
#ENV_NAME = "LunarLander-v2"
ENV_NAME = "CartPole-v0"
TRAIN_ITERATIONS = 1000
MAX_EPISODE_LENGTH = 1000
TRAJECTORY_BUFFER_SIZE = 32
BATCH_SIZE = 16
RENDER_EVERY = 100


if __name__ == "__main__":
    env = gym.make(ENV_NAME)
    agent = Agent(env.action_space.n, env.observation_space.shape, BATCH_SIZE)
    samples_filled = 0

    for cnt_episode in range(TRAIN_ITERATIONS):
        s = env.reset()
        r_sum = 0

        for cnt_step in range(MAX_EPISODE_LENGTH):
            '''
            if cnt_episode % RENDER_EVERY == 0:
                env.render()
            '''

            a = agent.choose_action(s)

            s_, r, done, _ = env.step(a)
            r /= 100
            r_sum += r
            if done:
                r = -1
            agent.store_transition(s, a, s_, r, done)
            samples_filled += 1

            #train in batches one buffer in filled with samples.
            if samples_filled % TRAJECTORY_BUFFER_SIZE == 0 and samples_filled != 0:
                for _ in range(TRAJECTORY_BUFFER_SIZE // BATCH_SIZE):
                    agent.train_network()
                agent.memory.clear()
                samples_filled = 0
            #set state to next state
            s = s_
            if done:
                break
        if cnt_episode % 10 == 0:
            print(f'Episode:{cnt_episode}, step:{cnt_step}, r_sum:{r_sum}')





