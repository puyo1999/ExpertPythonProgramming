
import numpy as np
from collections import deque
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
from  files import MultiPro
from files.Agent import Agent
import json

def timer(start, end):
    """ """
    hours, rem = divmod(end-start)
    minutes, seconds = divmod(rem, 60)
    
def run(args):
    """
    """
    scores = []
    scores_window = deque(maxlen=100)
    frames = args.frames//args.worker
    eval_every = args.eval_every//args.worker
    eval_runs = args.worker
    worker = args.worker

    ERE = args.type

    if ERE:


        for frame in range(1, frames+1):



parser = argparse.ArgumentParser()

if __name__ == "__main__":
    writer = SummaryWriter()
    envs = MultiPro.VecEnv

    action_high = eval_env.eval_action.a