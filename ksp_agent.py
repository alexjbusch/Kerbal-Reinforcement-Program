import math
import time
import krpc

import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
#matplotlib.use('TkAgg')
plt.ion()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

import utils



BATCH_SIZE = 128
LR = 1e-2
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.2
EPS_DECAY = 20000
TAU = 0.005

loss_function = nn.MSELoss()
#loss_function = nn.SmoothL1Loss()

num_episodes = 1e9
num_episodes = int(num_episodes)

handling_sensativity = 0.2
throttle_sensativity = 0.2

max_altitude = 20000
max_velocity = 200


episode_durations = []
episode_rewards = []
round_reward = 0

steps_done = 0
landed_counter = 0
current_epsilon = EPS_START

observations = ["yaw","pitch","roll","altitude","fuel",
                 "angular_velocity_x","angular_velocity_y","angular_velocity_z",
                 "velocity_x","velocity_y","velocity_z"]
actions = ["yaw_up","yaw_down","pitch_up","pitch_down","roll_up","roll_down","throttle_up","throttle_down","do_nothing"]
action_space = [action_int for action_int in range(len(actions))]
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



num_actions = len(actions)
num_observations= len(observations)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, num_observations, num_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(num_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, num_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

policy_net = DQN(num_observations, num_actions).to(device)
target_net = DQN(num_observations, num_actions).to(device)
#policy_net.load_state_dict(torch.load('saved_models/policy_net_epoch_'))
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)





conn = krpc.connect(name='ksp_agent')
vessel = conn.space_center.active_vessel

num_ship_parts = len(vessel.parts.all)

#vessel.control.input_mode = conn.space_center.ControlInputMode.override


def plot_rewards(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 10 episode averages and plot them too
    if len(rewards_t) >= 10:
        means = rewards_t.unfold(0, 10, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(9), means))
        plt.plot(means.numpy())
    
    plt.pause(1)  # pause a bit so that plots are updated
    #plt.show()
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())



def get_state():
    if vessel:
        yaw, pitch, roll = utils.get_yaw_pitch_roll(vessel, conn)
        fuel = vessel.resources_in_decouple_stage(vessel.control.current_stage-1).amount("LiquidFuel")
        throttle = vessel.control.throttle
        
        body_ref_frame = vessel.orbit.body.non_rotating_reference_frame
        if body_ref_frame:
            angular_velocity = vessel.angular_velocity(body_ref_frame)
            velocity = vessel.velocity(body_ref_frame)
        else:
            anguar_velocity = 0
            velocity = 0
        altitude = vessel.flight().surface_altitude

        state = torch.FloatTensor([yaw,pitch,roll,altitude,fuel,*angular_velocity,*velocity]).to(device)

    else:
        state = torch.zeroes(num_observations).to(device)

    return state



def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    global current_epsilon
    current_epsilon = eps_threshold

    """
    if random.randrange(0,100) < 1: 
        print(eps_threshold)
    """

    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            selected_action = policy_net(state).max(0)[1].view(1, 1)
    else:
        selected_action = torch.tensor([random.sample(action_space, 1)], device=device, dtype=torch.long)


    return selected_action


def do_action(action):
    match action:
        case 0:
            vessel.control.yaw += handling_sensativity
        case 1:
            vessel.control.yaw += -handling_sensativity
        case 2:
            vessel.control.pitch += handling_sensativity
        case 3:
            vessel.control.pitch += -handling_sensativity
        case 4:
            vessel.control.roll += handling_sensativity
        case 5:
            vessel.control.roll += -handling_sensativity
        case 6:
            vessel.control.throttle += throttle_sensativity
        case 7:
            vessel.control.throttle += -throttle_sensativity
        case 8:
            pass



def optimize_model():

    if len(memory) < BATCH_SIZE:
        return
    batch = memory.sample(BATCH_SIZE)
        
    state_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)
    #print(f"state_batch: {state_batch}")
    #print(f"reward_batch: {reward_batch}")
    #print(f"next_state_batch: {next_state_batch}")
    #print(f"terminal_batch: {terminal_batch}")
    state_batch = torch.stack(tuple(state for state in state_batch))


    
    reward_batch = torch.stack(reward_batch)
    next_state_batch = torch.stack(tuple(state for state in next_state_batch))

    if torch.cuda.is_available():
        state_batch = state_batch.cuda()
        reward_batch = reward_batch.cuda()
        next_state_batch = next_state_batch.cuda()

    q_values = policy_net(state_batch)
    policy_net.eval()
    
    with torch.no_grad():            
        next_prediction_batch = target_net(next_state_batch)
    #target_net.train()

    y_batch = torch.cat(
        tuple(reward if terminal else reward + GAMMA * prediction for reward, terminal, prediction in
              zip(reward_batch, terminal_batch, next_prediction_batch)))

    optimizer.zero_grad()
    #print(q_values.shape, y_batch.shape)
    loss = loss_function(q_values, y_batch.float())
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()



def get_reward(state):
    terminal = False


    parts_destroyed = num_ship_parts - len(vessel.parts.all)

    #reward = parts_destroyed * -20
    if vessel.situation == conn.space_center.VesselSituation.landed:
        global landed_counter
        landed_counter += 1
        if landed_counter > 5:
            terminal = True
            reward = 1000

    if parts_destroyed > 0:
        terminal = True
        reward = -100

    
    if not terminal:
        state_variables = {key:value.item() for key,value in zip(observations,state)}
        velocity_vector = [state_variables["velocity_x"],state_variables["velocity_y"],state_variables["velocity_z"]]
        velocity = utils.list_magnitude(velocity_vector)
        altitude = state_variables["altitude"]

        velocity_loss = velocity / max_velocity
        altitude_loss = altitude / max_altitude

        velocity_reward = 1 - velocity_loss
        altitude_reward = 1 - altitude_loss

        #reward = ((velocity_reward) + (altitude_reward)) / 2

        reward = velocity_reward
        
        if reward < 0.1:
            reward = -100
            terminal = True
        

    
    return reward, terminal


def update_policy_net():
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)


policy_net.load_state_dict(torch.load('saved_models/policy_net_epoch_50'))

conn.space_center.load('5k_mun_falling')
num_ship_parts = len(vessel.parts.all)
for i_episode in range(num_episodes):
    if i_episode % 50 == 0 or i_episode == 0:
        torch.save(policy_net.state_dict(), f"saved_models\\policy_net_epoch_{i_episode}")
    
    for t in count():
        state = get_state()
        action = select_action(state)
        do_action(action)

        next_state = get_state()

            
        reward, terminal = get_reward(next_state)
        print(f"reward: {reward}   eps: {current_epsilon}")
        round_reward += reward
        reward = torch.tensor([reward], device=device)
        # Store the transition in memory


        memory.push(state, action, next_state, reward)
        
        optimize_model()
        update_policy_net()

        if terminal:
            episode_rewards.append(round_reward)
            plot_rewards()
            terminal = False
            #plt.show()
            round_reward = 0.0
            landed_counter = 0
            break
    conn.space_center.load('10k_mun_falling')
