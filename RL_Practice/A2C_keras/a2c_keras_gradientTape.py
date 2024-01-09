from env_reinforce import CarrierStorage
from env_reinforce import Action
import random
from collections import defaultdict
import numpy as np
from termcolor import colored
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import copy
from keras.models import model_from_json
from collections import deque
from keras import backend as K
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

# custom loss를 구하기 위해 tensor를 즉시 확인.
import tensorflow as tf

tf.config.run_functions_eagerly(True)


# 여기 참조
# https://github.com/keras-team/keras-io/blob/master/examples/rl/actor_critic_cartpole.py

class A2CAgent(object):

    def __init__(self):

        # 단순하게 했을 경우에는 40으로 사용.
        self.state_size = 40  # float value 하나 사용
        self.action_size = 7

        self.discount_factor = 0.8

        self.DEFINE_NEW = False
        self.RENDER = False

        # self.actor = self.build_actor()
        # self.critic = self.build_critic()
        self.model = self.build_actorCritic()

    def build_actorCritic(self):
        if (self.DEFINE_NEW == True):
            input = Input(shape=(self.state_size,))
            common = Dense(self.state_size * 24, activation='relu', kernel_initializer='he_uniform')(input)
            common2 = Dense(self.action_size * 12, activation='relu', kernel_initializer='he_uniform')(common)
            action_prob = Dense(self.action_size, activation='softmax', kernel_initializer='he_uniform')(common2)
            critic = Dense(1)(common2)
            model = Model(inputs=input, outputs=[action_prob, critic])

        else:
            # 있는 데이터 로딩
            json_actor = open("./201027ActorA2c.json", "r")
            loaded_actor = json_actor.read()
            json_actor.close()
            model = model_from_json(loaded_actor)
            print("모델 %s를 로딩" % json_actor)
            weight_actor = "./201027weightCriticA2c.h5"
            model.load_weights(weight_actor)
            print("저장된 weights %s를 로딩" % weight_actor)
        return model

    def get_action(self, action_prob):
        # [[확율 형식으로 출력]]
        # [0]을 넣어 줌
        # print("policy = ", policy)
        return np.random.choice(self.action_size, 1, p=np.squeeze(action_prob))[0]


if __name__ == '__main__':

    # 메인 함수
    env = CarrierStorage()
    agent = A2CAgent()
    state = env.reset()

    # state history를 기록
    # historyState = []

    scores, episodes, score_average = [], [], []
    EPISODES = 100000

    global_step = 0
    average = 0
    huber_loss = tf.losses.Huber()
    optimizer = Adam(learning_rate=0.0001)

    # action, critic, reward를 list로 기록.
    actionprob_history, critic_history, reward_history = [], [], []

    for e in range(EPISODES):
        # print("episode check", e)
        done = False
        score = 0
        state = env.reset()
        state = env.stateTo1hot(agent.state_size)
        status = env.isItEnd()
        # print("reseted")
        if (status == 0 or status == 1):
            done = True
            reward = 0
            # print("zero rewards")
            # 여기에서 apply.gradients를 적용한면 안됨.
        while not done:
            if (agent.RENDER == True):
                env.render()
            global_step += 1
            # tape 아래로 모델을 입력해야 input, output 관계를 알 수 있음.
            # actor, critic 모두 예측.

            # with tf.GradientTape(persistent=True) as tape:
            with tf.GradientTape() as tape:
                action_prob, critic = agent.model(state)

                # action은 action tf.Tensor(
                # [[0.16487105 0.0549401  0.12524831 0.1738248  0.31119537 0.07012787  0.0997925 ]], shape=(1, 7), dtype=float32)
                # critic은
                # critic tf.Tensor([[0.04798129]], shape=(1, 1), dtype=float32)
                # 으로 출력.
                # action_prob로 action을 구함.
                action = agent.get_action(action_prob[0])
                # print("actionprob history",actionprob_history)
                if (agent.RENDER == True):
                    print("action is", Action(action))
                next_state, reward, done, info = env.step(action)

                # history에 추가
                critic_history.append(critic[0, 0])
                actionprob_history.append(tf.math.log(action_prob[0, action]))
                reward_history.append(reward)
                next_state = env.stateTo1hot(agent.state_size)
                # _, next_critic = agent.model(next_state)
                score += reward
                average = average + score
                state = copy.deepcopy(next_state)

                # rewards 를 discounted factor로 다시 계산.
                returns = []
                discounted_sum = 0
                for r in reward_history[::-1]:
                    discounted_sum = r + agent.discount_factor * discounted_sum
                    returns.insert(0, discounted_sum)

                # Normalize
                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
                returns = returns.tolist()

                # Calculating loss values to update our network
                history = zip(actionprob_history, critic_history, returns)
                actor_losses = []
                critic_losses = []
                for log_prob, value, ret in history:
                    advantage = ret - value
                    # advantage = reward  + (1.0 - done) * agent.discount_factor * next_critic - critic
                    # [ [prob, prob, ... ] ]형식으로 입력이 들어옮
                    actor_losses.append(-log_prob * advantage)
                    # critic_losses.append(advantage**2)
                    critic_losses.append(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))
                    # print("actor loss ", actor_losses)
                    # print("critic loss ", critic_losses)
                    # 모델이 하나라 actor_loss + critic_loss 더해서 한번에 train
                    # print("grad" , grads)
                    # print("history", len(actionprob_history))

                total_loss = actor_losses + critic_losses
                # loss도 gradientTape 안에 들어있어야 함.
            if (len(actionprob_history) > 0):
                # print("actor losses", len(actor_losses))
                # print("critic losses", len(critic_losses))
                # print("check", len(total_loss))
                grads = tape.gradient(total_loss, agent.model.trainable_weights)
                # print("grads", grads)
                optimizer.apply_gradients(zip(grads, agent.model.trainable_weights))
                # print("actionprob history", actionprob_history)
                # print("cirtic,",critic_history)
                # print("rewards", reward_history)
                # print("actor losses", len(actor_losses))
                # print("critic losses", len(critic_losses))
                # print("total loss", len(total_loss))

                # print("actionprob_history", len(actionprob_history))
                # print("episodes", e)
        if (agent.RENDER == True):
            print("episode:", e, "  score:", score)
        if (e % 100 == 0):
            print("history length is", len(actionprob_history))
            print("episode:", e, "  score:", score, "global_step", global_step, "average", average)
            scores.append(score)
            score_average.append(average)
            episodes.append(e)
            # 매 1000회마다 average 초기화.
            average = 0
            model_json_actor = agent.model.to_json()
            with open("./201027ActorA2c.json", "w") as json_file:
                json_file.write(model_json_actor)
            agent.model.save_weights("./201027weightCriticA2c.h5")
            plt.plot(episodes, score_average, 'b')
            # plt.show()
            plt.savefig("./history.png")
            # 비어있는 history로 gradients를 계산하지 않도록..
            # print("episode", e)
            actionprob_history.clear()
            critic_history.clear()
            reward_history.clear()

    plt.plot(episodes, score_average, 'b')
    # plt.show()
    plt.savefig("./history.png")