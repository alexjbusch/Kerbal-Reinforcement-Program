import math
import random
import matplotlib.pyplot as plt
import torch
import utils

from IPython import display

from hyperparameters import BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU
from formulation import OBS, HANDLING_SENSITIVITY, THROTTLE_SENSITIVITY, MAX_ALTITUDE, MAX_VELOCITY
from Transition import Transition


class Game:

    def __init__(self, conn, episode_rewards, device, num_observations, memory, policy_net, target_net, action_space,
                 optimizer, loss_function, show_result=False):
        self.conn = conn
        self.episode_rewards = episode_rewards
        self.show_result = show_result
        self.device = device
        self.num_observations = num_observations
        self.memory = memory
        self.policy_net = policy_net
        self.target_net = target_net
        self.action_space = action_space
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.vessel = conn.space_center.active_vessel
        self.steps_done = 0
        self.current_epsilon = EPS_START
        self.landed_counter = 0
        self.round_reward = 0
        self.num_ship_parts = len(self.vessel.parts.all)

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
                                   *velocity,
                                   *rotation]).to(self.device)

            # print(state)

        else:
            s = torch.zeros(self.num_observations).to(self.device)

        return s

    def select_action(self, s):
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

    def do_action(self, a):
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
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s[None] for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        state_batch = state_batch.view(-1, self.num_observations)

        assert torch.equal(batch.state[0], state_batch[0])

        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = self.loss_function
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

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

            if new_reward < -.9:
                new_reward = -100
                is_terminal = True

        return new_reward, is_terminal

    def update_policy_net(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        self.target_net.load_state_dict(target_net_state_dict)
