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
from hyperparameters import MODEL_TO_LOAD, LR, NUM_EPISODES
from formulation import OBS, ACTIONS
from SaveAction import SaveAction
from Game import Game
import time
from itertools import count


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

actor_critic_net = ActorCritic(len(OBS),len(ACTIONS), device)
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
            SaveAction=SaveAction,
            actor_critic_model = actor_critic_net,
            action_space=[action_int for action_int in range(len(ACTIONS))],
            optimizer=optimizer,
            loss_function=nn.SmoothL1Loss(),
            show_result=False)

def reset():
    # terminal = False
    game.landed_counter = 0
    game.episode_rewards.append(game.current_round_reward)
    game.current_round_reward = 0
    game.plot_rewards()

def main():
    plt.ion()
    running_reward = 10
    
    # used to tally up the total amount of states the model has trained on
    frames_seen = 0


    game.load()
    game.num_ship_parts = len(game.vessel.parts.all)

    for i_episode in range(NUM_EPISODES):
        start = time.time()

        game.load()
        for frame in count():
    
            state = game.get_state()
            action = game.select_action(state)
            game.do_action(action)
            time.sleep(0.05)
            next_state = game.get_state()
            reward, terminal = game.get_reward(next_state)

            game.actor_critic_model.rewards.append(reward)
            game.current_round_reward += reward
            

            #if frames_seen % 30 == 0:
                #print(f"reward: {reward} )

            frames_seen += 1
            
            end = time.time()
            if end-start > 45:
                terminal = True
                reward = -200
                
            if terminal:
                print("ROUND ENDED")
                next_state = None
                terminal = False
                break
        

        # update cumulative reward
        # running_reward = 0.05 * game.round_reward + (1 - 0.05) * running_reward
        running_reward = 0.05 * game.current_round_reward + (1 - 0.05) * running_reward

            

        if i_episode:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, game.current_round_reward, running_reward))
        game.optimize_model()
        reset()

        

if __name__ == '__main__':
    main()
plt.ioff()
plt.show()