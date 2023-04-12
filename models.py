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
    def __init__(self, state_size, action_size, device):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # self.affine1 = nn.Linear(self.state_size, 128)

        # actor's layer
        self.actor = nn.Sequential(
               nn.Linear(self.state_size, 512),
               nn.ReLU(),
               nn.Linear(512, 512),
               nn.ReLU(),
               nn.Linear(512, self.action_size)
               )
        

        # critic's layer
        self.critic =  nn.Sequential(
               nn.Linear(self.state_size, 128),
               nn.ReLU(),
               nn.Linear(128, 128),
               nn.ReLU(),
               nn.Linear(128, 1)
               )


        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

        self.to(device)

    def forward(self,x):
        """
        forward of both actor and critic
        """

        # actor: choses action to take from state s_t
        # by returning probability of each action
        actor = F.softmax(self.actor(x), dim=-1)

        # critic: evaluates being in the state s_t
        critic = self.critic(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return actor, critic
