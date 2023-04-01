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

model_to_load = ''


BATCH_SIZE = 32
LR = 1e-4
GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.1
EPS_DECAY = 1000
TAU = 0.005
REPLAY_MEMORY_SIZE = 10000

#loss_function = nn.MSELoss()
loss_function = nn.SmoothL1Loss()

num_episodes = 1e9
num_episodes = int(num_episodes)

handling_sensativity = 0.25
throttle_sensativity = 0.3

max_altitude = 20000
max_velocity = 200


episode_durations = []
episode_rewards = []
round_reward = 0

steps_done = 0
landed_counter = 0
current_epsilon = EPS_START

observations = ["angle_of_attack","sideslip_angle","altitude","fuel",
                 "angular_velocity_x","angular_velocity_y","angular_velocity_z",
                 "velocity_x","velocity_y","velocity_z",
                "rotation_x","rotation_y","rotation_z","rotation_w"]
actions = ["yaw_up","yaw_down","pitch_up","pitch_down","roll_up","roll_down","throttle_up","throttle_down","do_nothing"]
action_space = [action_int for action_int in range(len(actions))]
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



num_actions = len(actions)
num_observations= len(observations)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)



policy_net = DQN(num_observations, num_actions).to(device)
target_net = DQN(num_observations, num_actions).to(device)

"""
if model_to_load not in {None, ""}:
    policy_net.load_state_dict(torch.load(model_to_load))
"""
#target_net.load_state_dict(policy_net.state_dict())


optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)






conn = krpc.connect(name='ksp_agent')
vessel = conn.space_center.active_vessel

#vessel.control.input_mode = conn.space_center.ControlInputMode.override






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
    
memory = ReplayMemory(REPLAY_MEMORY_SIZE)



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
    
    plt.pause(0.5)  # pause a bit so that plots are updated
    #plt.show()
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())



def get_state():
    if vessel:
        #antiradial, prograde, normal = vessel.direction(vessel.orbital_reference_frame)
        fuel = vessel.resources_in_decouple_stage(vessel.control.current_stage-1).amount("LiquidFuel")
        throttle = vessel.control.throttle

        ref_frame = vessel.orbit.body.reference_frame
        flight = vessel.flight(ref_frame)
        
        angle_of_attack = flight.angle_of_attack
        sideslip_angle = flight.sideslip_angle


        angular_velocity = vessel.angular_velocity(ref_frame)
        velocity = vessel.velocity(ref_frame)
        rotation = flight.rotation
        altitude = flight.surface_altitude

        #state = torch.FloatTensor([prograde, antiradial, normal ,altitude,fuel,*angular_velocity,*velocity]).to(device)
        state = torch.FloatTensor([angle_of_attack,sideslip_angle,altitude,fuel,*angular_velocity,*velocity, *rotation]).to(device)

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



Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)


    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))



    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)


    non_final_next_states = torch.cat([s[None] for s in batch.next_state
                                                if s is not None])


    #import pdb; pdb.set_trace()
    #print(non_final_next_states.shape)
    #print(non_final_next_states)

    state_batch = torch.cat(batch.state, dim=0)
    state_batch = state_batch.view(-1,num_observations)
    
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net

    #print(f"action_batch: {action_batch.shape}")
    #64x1
    #print(f"state_batch: {state_batch.shape}")
    #64x11
    state_action_values = policy_net(state_batch).gather(1, action_batch)



    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def get_reward(state):
    terminal = False


    parts_destroyed = num_ship_parts - len(vessel.parts.all)

    #reward = parts_destroyed * -20
    global landed_counter
    if vessel.situation == conn.space_center.VesselSituation.landed:   
        landed_counter += 1
        if landed_counter > 5:
            terminal = True
            reward = 1000
    else:
        landed_counter = 0

    if parts_destroyed > 0:
        terminal = True
        reward = -100

    
    if not terminal:
        state_variables = {key:value.item() for key,value in zip(observations,state)}
        velocity_vector = [state_variables["velocity_x"],state_variables["velocity_y"],state_variables["velocity_z"]]
        velocity = utils.list_magnitude(velocity_vector)

        angular_velocity_vector = [state_variables["angular_velocity_x"],state_variables["angular_velocity_y"],state_variables["angular_velocity_z"]]
        angular_velocity = utils.list_magnitude(angular_velocity_vector)
        reward = angular_velocity

        
        altitude = state_variables["altitude"]

        
        velocity_loss = velocity / max_velocity
        altitude_loss = altitude / max_altitude
        

        velocity_reward = 1 - velocity_loss
        altitude_reward = 1 - altitude_loss
        #pitch_reward = -state_variables["prograde"]

        reward = ((velocity_reward) + (altitude_reward)) / 2
        #reward = velocity_reward
        #reward = (reward + 1)**10
        #reward = pitch_reward
        
        if reward < 0.4:
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







# max was 500k
frames_seen = 0

for i_episode in range(num_episodes):
    if i_episode % 25 == 0 or i_episode == 0:
        torch.save(policy_net.state_dict(), f"saved_models\\policy_net_epoch_{i_episode}")
    conn.space_center.load('10k_mun_falling')
    # this must be called AFTER the save is loaded or the num_parts will be 0
    num_ship_parts = len(vessel.parts.all)

    vessel.control.sas_mode = conn.space_center.SASMode.retrograde

    for t in count():
        frames_seen += 1
        
        state = get_state()
        action = select_action(state)
        do_action(action)
        time.sleep(0.05)
        next_state = get_state()
        reward, terminal = get_reward(next_state)



        
        if frames_seen % 10 == 0:
            print(f"reward: {reward}   eps: {current_epsilon}, frame: {round(frames_seen/1000000, 5)}M")
        round_reward += reward
        reward = torch.tensor([reward], device=device)
        # Store the transition in memory

        if terminal:
            next_state = None
        memory.push(state, action, next_state, reward)
        
        optimize_model()
        update_policy_net()

        if terminal:
            episode_rewards.append(round_reward)
            plot_rewards()
            terminal = False
            plt.show()
            round_reward = 0.0
            landed_counter = 0
            break
    
