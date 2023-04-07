# Kerbal-Reinforcement-Program
A Deep Q Learning implementation for landing rockets in Kerbal Space Program 1

This project aims to use the Deep Q Learning algorithm to train an AI agent to autonomously land rockets in KSP.  It uses pytorch for the neural network and the kRPC mod to control the rocket from the python script.

## Navigation
`formulation.py`: contains information about the problem formulation (action space, observations, handling sensitivity, etc.)

`hyperparameters.py`: holds the hyperparameters for DQN (batch size, learning rate, tau, gamma, etc.)

`models.py`: defines our DQN model

`utils.py`: has simple functions that are useful for DQNs

`ksp_agent.py`: contains the main training loop for our agent

`Transition.py`: defines the transition class

`ReplayMemory.py`: defines the ReplayMemoryClass

`Game.py`: defines the class which wraps the functions and parameters needed to train an agent into one place


