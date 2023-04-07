import time
import krpc
import sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import os

from models import DQN
from hyperparameters import MODEL_TO_LOAD, LR, REPLAY_MEMORY_SIZE, NUM_EPISODES
from formulation import OBS, ACTIONS
from ReplayMemory import ReplayMemory
from Game import Game


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(len(OBS), len(ACTIONS)).to(device)
target_net = DQN(len(OBS), len(ACTIONS)).to(device)

if MODEL_TO_LOAD not in [None, ""]:
    policy_net.load_state_dict(torch.load(MODEL_TO_LOAD))

conn = krpc.connect(name='ksp_agent')
vessel = conn.space_center.active_vessel

game = Game(conn=conn,
            episode_rewards=[],
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            num_observations=len(OBS),
            memory=ReplayMemory(REPLAY_MEMORY_SIZE),
            policy_net=policy_net,
            target_net=target_net,
            action_space=[action_int for action_int in range(len(ACTIONS))],
            optimizer=optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True),
            loss_function=nn.SmoothL1Loss(),
            show_result=False)
frames_seen = 0

for i_episode in range(NUM_EPISODES):
    if i_episode % 25 == 0 or i_episode == 0:
        if not os.path.exists('saved_models'):
            os.makedirs('saved_models')
        if sys.platform == "win32":
            torch.save(game.policy_net.state_dict(), f"saved_models\\policy_net_epoch_{i_episode}")
        else:
            torch.save(game.policy_net.state_dict(), "saved_models/policy_net_epoch_{0}".format(i_episode))
    game.conn.space_center.load('5k_mun_falling')
    game.num_ship_parts = len(game.vessel.parts.all)

    for t in itertools.count():
        frames_seen += 1

        state = game.get_state()
        action = game.select_action(state)
        game.do_action(action)
        time.sleep(0.05)
        next_state = game.get_state()
        reward, terminal = game.get_reward(next_state)

        if frames_seen % 10 == 0:
            print(f"reward: {reward}   eps: {game.current_epsilon}, frame: {round(frames_seen / 1000000, 5)}M")
        game.round_reward += reward
        reward = torch.tensor([reward], device=game.device)

        if terminal:
            next_state = None
        game.memory.push(state, action, next_state, reward)

        game.optimize_model()
        game.update_policy_net()

        if terminal:
            game.episode_rewards.append(game.round_reward)
            terminal = False
            game.round_reward = 0.0
            game.landed_counter = 0
            # plot_rewards()
            # plt.show()
            break

game.plot_rewards()
plt.show()
