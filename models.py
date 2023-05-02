import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch


# In the Actor-Critic method, the policy is referred to as the actor that 
# proposes a set of possible actions given a state, and the estimated value 
# function is referred to as the critic, which evaluates actions taken by the 
# actor based on the given policy.

# 
 
def add_noise (x):
    return  x + (0.1**0.5) * torch.randn_like (x)

class ActorCritic(nn.Module):
    """
    implements both actor and critic in one model, there is only one layer
    """
    def __init__(self, state_size, action_size, device):
        super(ActorCritic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # self.affine1 = nn.Linear(self.state_size, 128)
        self.size = 256
        self.lin1 = torch.nn.Linear (in_features=self.action_size, out_features=self.action_size)
        
        # actor's layer
        self.actor = nn.Sequential(
               nn.Linear(self.state_size, self.size),
               nn.LeakyReLU(0.1),
               nn.Linear(self.size, self.size),
               nn.LeakyReLU(0.1),
               nn.Linear(self.size, self.action_size)
               )
        

        # critic's layer
        self.critic =  nn.Sequential(
               nn.Linear(self.state_size, self.size),
               nn.LeakyReLU(0.1),
               nn.Linear(self.size, self.size),
               nn.LeakyReLU(0.1),
               nn.Linear(self.size, 1)
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
        actor_out = self.actor(x)
        actor_out_with_noise = torch.nn.functional.linear (actor_out, add_noise (self.lin1.weight), self.lin1.bias)
        actor = F.softmax( actor_out_with_noise, dim=-1)

        # critic: evaluates being in the state s_t
        critic = self.critic(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return actor, critic
