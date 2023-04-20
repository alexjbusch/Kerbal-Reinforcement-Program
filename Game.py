import math
import random
import matplotlib.pyplot as plt
import torch
from torch.distributions import Categorical
import utils
from numpy import finfo, float32
import torch.nn.functional as F


from IPython import display

from hyperparameters import BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU
from formulation import OBS, HANDLING_SENSITIVITY, THROTTLE_SENSITIVITY, MAX_ALTITUDE, MAX_VELOCITY


class Game:

    def __init__(self, conn, episode_rewards, device, num_observations, SaveAction, actor_critic_model, action_space,
                 optimizer, loss_function, show_result=False):
        self.conn = conn
        self.episode_rewards = episode_rewards
        self.show_result = show_result
        self.device = device
        self.num_observations = num_observations
        self.SaveAction = SaveAction
        self.actor_critic_model = actor_critic_model
        self.action_space = action_space
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.vessel = conn.space_center.active_vessel
        self.steps_done = 0
        self.current_epsilon = EPS_START
        self.landed_counter = 0
        self.ep_reward = 0
        self.num_ship_parts = len(self.vessel.parts.all)

        self.machine_epsilon = finfo(float32).eps.item()

    def plot_rewards(self):
        plt.figure(1)
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        if self.show_result:
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

        plt.pause(0.5)
        display.display(plt.gcf())

    def get_state(self):
        if self.vessel:
            # antiradial, prograde, normal = vessel.direction(vessel.orbital_reference_frame)
            # fuel = vessel.resources_in_decouple_stage(vessel.control.current_stage-1).amount("LiquidFuel")

            throttle = self.vessel.control.throttle

            ref_frame = self.vessel.orbit.body.reference_frame
            flight = self.vessel.flight(ref_frame)

            # angle_of_attack = flight.angle_of_attack
            # sideslip_angle = flight.sideslip_angle
            # angular_velocity = vessel.angular_velocity(ref_frame)

            velocity = self.vessel.velocity(ref_frame)
            rotation = flight.rotation
            altitude = flight.surface_altitude

            # state = torch.FloatTensor([prograde, antiradial, normal ,altitude,fuel,
            #                            *angular_velocity,*velocity]).to(device)
            # state = torch.FloatTensor([angle_of_attack,sideslip_angle,altitude,fuel,
            #                            *angular_velocity,*velocity, *rotation]).to(device)
            s = torch.FloatTensor([throttle, altitude,
                                   *velocity,*rotation
                                   ]).to(self.device)

            # print(state)

        else:
            s = torch.zeros(self.num_observations).to(self.device)

        return s

    def select_action(self, state):
        probs, state_value = self.actor_critic_model(state)
        # print("probs before: ", probs.weight)
        # noise = torch.zeros(9, dtype=torch.float64)
        # noise = noise + (0.1**0.5) * torch.randn(9)
        # probs = probs + noise
        print("probs after: ", probs)
        

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)
        

        # and sample an action using the distribution
        print("m", m)
        action = m.sample()
        print("action", action)

        # save to action buffer
        self.actor_critic_model.saved_actions.append(self.SaveAction(m.log_prob(action), state_value))
        # the action to take (left or right)

        return action.item()

        """
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.current_epsilon = eps_threshold

        # if random.randrange(0,100) < 1:
        #     print(eps_threshold)

        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                selected_action = self.policy_net(s).max(0)[1].view(1, 1)
        else:
            selected_action = torch.tensor([random.sample(self.action_space, 1)], device=self.device, dtype=torch.long)

        return selected_action
        """

    def do_action(self, a):
        #print(f"action selected: {a}")
        # match a:
        #     case 0:
        #         self.vessel.control.throttle += THROTTLE_SENSITIVITY
        #     case 1:
        #         self.vessel.control.throttle -= THROTTLE_SENSITIVITY
        
        match a:
            case 0:
                self.vessel.control.yaw += HANDLING_SENSITIVITY
            case 1:
                self.vessel.control.yaw += -HANDLING_SENSITIVITY
            case 2:
                self.vessel.control.pitch += HANDLING_SENSITIVITY
            case 3:
                self.vessel.control.pitch += -HANDLING_SENSITIVITY
            case 4:
                self.vessel.control.roll += HANDLING_SENSITIVITY
            case 5:
                self.vessel.control.roll += -HANDLING_SENSITIVITY
            case 6:
                self.vessel.control.throttle += THROTTLE_SENSITIVITY
            case 7:
                self.vessel.control.throttle += -THROTTLE_SENSITIVITY
            case 8:
                pass
        

    def optimize_model(self):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = self.actor_critic_model.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.actor_critic_model.rewards[::-1]:
            # calculate the discounted value
            R = r + GAMMA * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.machine_epsilon)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        print("loss", loss)
        # perform backprop
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.actor_critic_model.rewards[:]
        del self.actor_critic_model.saved_actions[:]

        return loss

    def get_reward(self, s):
        is_terminal = False

        parts_destroyed = self.num_ship_parts - len(self.vessel.parts.all)

        # new_reward = parts_destroyed * -20
        new_reward = None
        if self.vessel.situation == self.conn.space_center.VesselSituation.landed:
            self.landed_counter += 1
            if self.landed_counter > 5:
                is_terminal = True
                new_reward = 1000
        else:
            self.landed_counter = 0

        if parts_destroyed > 0:
            is_terminal = True
            new_reward = -100

        if not is_terminal:
            state_variables = {key: value.item() for key, value in zip(OBS, s)}
            velocity_vector = [state_variables["velocity_x"], state_variables["velocity_y"],
                               state_variables["velocity_z"]]
            velocity = utils.list_magnitude(velocity_vector)

            # angular_velocity_vector = [state_variables["angular_velocity_x"],state_variables["angular_velocity_y"],
            #                            state_variables["angular_velocity_z"]]
            # angular_velocity = utils.list_magnitude(angular_velocity_vector)
            # new_reward = angular_velocity

            altitude = state_variables["altitude"]

            velocity_loss = velocity / MAX_VELOCITY
            altitude_loss = altitude / MAX_ALTITUDE

            velocity_reward = 1 - velocity_loss
            altitude_reward = 1 - altitude_loss

            # pitch_reward = -state_variables["prograde"]

            new_reward = (velocity_reward + altitude_reward) / 2
            # new_reward = velocity_reward
            # new_reward = (reward + 1)**10
            # new_reward = pitch_reward

            # if new_reward < 0:
            #     new_reward = -100
            #     is_terminal = True

        return new_reward, is_terminal

    def load(self):
        try:
            self.conn.space_center.load('10k_mun_falling')
        except ValueError:
            self.conn.space_center.load('10 k mun falling')
        

    

