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

        self.first_eps = True
        self.vessel = conn.space_center.active_vessel
        self.speedMode = self.vessel.control.speed_mode
        self.steps_done = 0
        self.prev_delta_alt = 5505
        self.current_epsilon = EPS_START
        self.landed_counter = 0
        self.ep_reward = 0
        self.num_ship_parts = len(self.vessel.parts.all)
        self.prev_vel = 0
        self.prev_alt = None
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

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)
        # print(probs)
        

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        self.actor_critic_model.saved_actions.append(self.SaveAction(m.log_prob(action), state_value))
        # the action to take (left or right)
        # print(action.item())

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
        match a:
            case 0:
                self.vessel.control.throttle += THROTTLE_SENSITIVITY
            case 1:
                self.vessel.control.throttle -= THROTTLE_SENSITIVITY
            case 2:
                pass
        
        # match a:
        #     # case 0:
        #     #     self.vessel.control.yaw += HANDLING_SENSITIVITY
        #     # case 1:
        #     #     self.vessel.control.yaw += -HANDLING_SENSITIVITY
        #     # case 2:
        #     #     self.vessel.control.pitch += HANDLING_SENSITIVITY
        #     # case 3:
        #     #     self.vessel.control.pitch += -HANDLING_SENSITIVITY
        #     # case 4:
        #     #     self.vessel.control.roll += HANDLING_SENSITIVITY
        #     # case 5:
        #     #     self.vessel.control.roll += -HANDLING_SENSITIVITY
        #     case 6:
        #         self.vessel.control.throttle += THROTTLE_SENSITIVITY
        #     case 7:
        #         self.vessel.control.throttle += -THROTTLE_SENSITIVITY
        #     case 8:
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
                new_reward = 10000
        else:
            self.landed_counter = 0

        if parts_destroyed > 0:
            is_terminal = True
            new_reward = -5000

        if not is_terminal:
            state_variables = {key: value.item() for key, value in zip(OBS, s)}
            velocity_vector = [state_variables["velocity_x"], state_variables["velocity_y"],
                               state_variables["velocity_z"]]
            velocity = utils.list_magnitude(velocity_vector)

            altitude = state_variables["altitude"]
            vessel_speed_relative_to_mun = self.vessel.orbit.speed
            velocity = vessel_speed_relative_to_mun
            acceleration = velocity - self.prev_vel
            self.prev_vel = vessel_speed_relative_to_mun
            

            velocity_reward = 10*(1/vessel_speed_relative_to_mun)**0.5

            if self.prev_alt == None:
                self.prev_alt = altitude
                distance_reward = (1/altitude)**0.5
            else:
                delta_altitude = (self.prev_alt - altitude)
                self.prev_delta_alt = delta_altitude
                self.prev_alt = altitude

                if delta_altitude < 0:
                    distance_reward = delta_altitude
                else:
                    distance_reward = delta_altitude * (1/altitude)**0.5


            new_reward = velocity_reward + distance_reward

            if altitude > 6000:
                is_terminal = True
                new_reward = -200
            
            if altitude > 2000:
                new_reward = velocity_reward + 40 * distance_reward
            else:
                new_reward = 20*velocity_reward + distance_reward
                
            # print("alt: ", altitude)
         
            # print("altitude change", delta_altitude)
        #     print("distance_reward",distance_reward)
        #     print("velocity_reward",velocity_reward)
        #     print('reward1', new_reward)
        # print("new_reward2 : ", new_reward)
        # print("")
            

           
         
                

        return new_reward, is_terminal

    def load(self):
        try:
            self.conn.space_center.load('10k_mun_falling')
        except ValueError:
            self.conn.space_center.load('10 k mun falling')
        

    

