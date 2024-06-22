import numpy as np
import tensorflow as tf
import gym
import tensorflow_probability as tfp
from multiprocessing import Process, Queue, Barrier, Lock
import keras.losses as kls

env = gym.make('CartPole-v0')
low = env.observation_space.low
high = env.observation_space.high

import yaml

import logging
logger = logging.getLogger(__name__)

with open('./spec/spec.yaml') as f:
    config = yaml.safe_load(f)