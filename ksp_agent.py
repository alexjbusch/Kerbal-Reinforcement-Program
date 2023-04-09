import time
import krpc
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import os
import numpy as np

from models import ActorCritic
from hyperparameters import MODEL_TO_LOAD, LR, REPLAY_MEMORY_SIZE, NUM_EPISODES
from formulation import OBS, ACTIONS
from ReplayMemory import ReplayMemory
from Game import Game
from math import count


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor_critic_net = ActorCritic()
optimizer = optim.Adam(actor_critic_net.parameters(), lr=LR, amsgrad=True)
eps = np.finfo(np.float32).eps.item()

if MODEL_TO_LOAD not in [None, ""]:
    actor_critic_net.load_state_dict(torch.load("./saved_models/policy_net_epoch_75"))

conn = krpc.connect(name='ksp_agent')
vessel = conn.space_center.active_vessel

game = Game(conn=conn,
            episode_rewards=[],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_observations=len(OBS),
            memory=ReplayMemory(REPLAY_MEMORY_SIZE),
            actor_critic_model = actor_critic_net,
            action_space=[action_int for action_int in range(len(ACTIONS))],
            optimizer=optimizer,
            loss_function=nn.SmoothL1Loss(),
            show_result=False)

def reset():
    game.episode_rewards.append(game.round_reward)
    terminal = False
    game.round_reward = 0.0
    game.landed_counter = 0
    game.plot_rewards()

def main():
    running_reward = 10

    # run infinitely many episodes
    for i_episode in range(NUM_EPISODES):
        print("start new episode")
        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0

if __name__ == '__main__':
    main()
