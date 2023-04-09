import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# In the Actor-Critic method, the policy is referred to as the actor that 
# proposes a set of possible actions given a state, and the estimated value 
# function is referred to as the critic, which evaluates actions taken by the 
# actor based on the given policy.

# 
    
class ActorCritic(nn.Module):
    """
    implements both actor and critic in one model, there is only one layer
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.affine1 = nn.Linear(self.state_size, 128)

        # actor's layer
        self.action_head = nn.Linear(128, self.action_size)

        # critic's layer
        self.value_head = nn.Linear(128, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, state_size, action_size):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(self.state_size), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values
