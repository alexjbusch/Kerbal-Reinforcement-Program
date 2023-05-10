import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch


# In the Actor-Critic method, the policy is referred to as the actor that 
# proposes a set of possible actions given a state, and the estimated value 
# function is referred to as the critic, which evaluates actions taken by the 
# actor based on the given policy.

# 

class ActorCritic(nn.Module):
  def __init__(self, state_size, action_size, device):
    super().__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.size = 256

    self.shared_layers = nn.Sequential(
        nn.Linear(state_size, 64),
        nn.LeakyReLU(0.1),
        nn.Linear(64, self.state_size),
        nn.LeakyReLU(0.1),
    )
    
    self.policy_layers = nn.Sequential(
        nn.Linear(self.state_size, self.size),
        nn.LeakyReLU(0.1),
        nn.Linear(self.size, self.size),
        nn.LeakyReLU(0.1),
        nn.Linear(self.size, self.action_size)
    )
    
    self.value_layers = nn.Sequential(
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
    
  def value(self, obs):
    z = self.shared_layers(obs)
    value = self.value_layers(z)
    return value
        
  def policy(self, obs):
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    return policy_logits

  def forward(self, obs):
    z = self.shared_layers(obs)
    policy_logits = self.policy_layers(z)
    value = self.value_layers(z)
    return policy_logits, value
 
class PPOTrainer():
  def __init__(self,
              actor_critic,
              ppo_clip_val=0.2,
              target_kl_div=0.01,
              max_policy_train_iters=80,
              value_train_iters=80,
              policy_lr=3e-4,
              value_lr=1e-3):
    self.ac = actor_critic
    self.ppo_clip_val = ppo_clip_val
    self.target_kl_div = target_kl_div
    self.max_policy_train_iters = max_policy_train_iters
    self.value_train_iters = value_train_iters

    policy_params = list(self.ac.shared_layers.parameters()) + \
        list(self.ac.policy_layers.parameters())
    self.policy_optim = torch.optim.Adam(policy_params, lr=policy_lr)

    value_params = list(self.ac.shared_layers.parameters()) + \
        list(self.ac.value_layers.parameters())
    self.value_optim = torch.optim.Adam(value_params, lr=value_lr)

  def train_policy(self, obs, acts, old_log_probs, gaes):
    for _ in range(self.max_policy_train_iters):
      self.policy_optim.zero_grad()

      new_logits = self.ac.policy(obs)
      new_logits = Categorical(logits=new_logits)
      new_log_probs = new_logits.log_prob(acts)

      policy_ratio = torch.exp(new_log_probs - old_log_probs)
      clipped_ratio = policy_ratio.clamp(
          1 - self.ppo_clip_val, 1 + self.ppo_clip_val)
      
      clipped_loss = clipped_ratio * gaes
      full_loss = policy_ratio * gaes
      policy_loss = -torch.min(full_loss, clipped_loss).mean()

      policy_loss.backward()
      self.policy_optim.step()

      kl_div = (old_log_probs - new_log_probs).mean()
      if kl_div >= self.target_kl_div:
        break
      return policy_loss

  def train_value(self, obs, returns):
    for _ in range(self.value_train_iters):
      self.value_optim.zero_grad()

      values = self.ac.value(obs)
      value_loss = (returns - values) ** 2
      value_loss = value_loss.mean()

      value_loss.backward()
      self.value_optim.step()

      return value_loss


# def add_noise (x):
#     return  x + (0.1**0.5) * torch.randn_like (x)

# class ActorCritic(nn.Module):
#     """
#     implements both actor and critic in one model, there is only one layer
#     """
#     def __init__(self, state_size, action_size, device):
#         super(ActorCritic, self).__init__()
#         self.state_size = state_size
#         self.action_size = action_size

#         # self.affine1 = nn.Linear(self.state_size, 128)
#         self.size = 256
#         self.lin1 = torch.nn.Linear (in_features=self.action_size, out_features=self.action_size)
        
#         # actor's layer
#         self.actor = nn.Sequential(
#                nn.Linear(self.state_size, self.size),
#                nn.Tanh(),
#                nn.Linear(self.size, self.size),
#                nn.Tanh(),
#                nn.Linear(self.size, self.action_size)
#                )
        

#         # critic's layer
#         self.critic =  nn.Sequential(
#                nn.Linear(self.state_size, self.size),
#                nn.Tanh(),
#                nn.Linear(self.size, self.size),
#                nn.Tanh(),
#                nn.Linear(self.size, 1)
#                )


#         # action & reward buffer
#         self.saved_actions = []
#         self.rewards = []

#         self.to(device)

#     def forward(self,x):
#         """
#         forward of both actor and critic
#         """

#         # actor: choses action to take from state s_t
#         # by returning probability of each action
#         actor_out = self.actor(x)
#         actor_out_with_noise = torch.nn.functional.linear (actor_out, add_noise (self.lin1.weight), self.lin1.bias)
#         actor = F.softmax( actor_out_with_noise, dim=-1)

#         # critic: evaluates being in the state s_t
#         critic = self.critic(x)

#         # return values for both actor and critic as a tuple of 2 values:
#         # 1. a list with the probability of each action over the action space
#         # 2. the value from state s_t
#         return actor, critic
