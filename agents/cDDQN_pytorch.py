import random
from collections import deque, defaultdict, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MINVALUE = -1000
TAU = 1e-3  # Soft update parameter for updating fixed q network

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        """
        Network for DDQN Agent
        
        Parameters
        ----------
        state_size (int): continuous state dimenion
        action_size (int): number of actions
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer:
    def __init__(self, max_buffer_size, device):
        """
        Experience Replay Buffer

        Parameters
        ----------
        max_buffer_size (int): maximum size of internal memory
        device (string): device (cuda:# or cpu)
        """
        
        self.memory = deque(maxlen=max_buffer_size)
        self.max_buffer_size = max_buffer_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "terminate"])
        self.device = device
        self.n_items = 0

    def add_experience(self, state, action, reward, next_state, terminate):
        """
        Add experience to buffer

        Parameters
        ----------
        state (array_like)
        action (int)
        reward (float)
        next_state (array_like)
        terminate (bool)
        """
        experience = self.experience(state, action, reward, next_state, terminate)
        self.memory.append(experience)
        self.n_items = min(self.n_items + 1, self.max_buffer_size)

    def sample_batch(self, batch_size):
        experiences = random.sample(self.memory, k=batch_size)
        states = torch.from_numpy(
            np.vstack([experience.state for experience in experiences if experience is not None])).float().to(
            self.device)
        actions = torch.from_numpy(
            np.vstack([experience.action for experience in experiences if experience is not None])).long().to(
            self.device)
        rewards = torch.from_numpy(
            np.vstack([experience.reward for experience in experiences if experience is not None])).float().to(
            self.device)
        next_states = torch.from_numpy(
            np.vstack([experience.next_state for experience in experiences if experience is not None])).float().to(
            self.device)
        # Convert terminate from boolean to int
        terminates = torch.from_numpy(
            np.vstack([experience.terminate for experience in experiences if experience is not None]).astype(
                np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, terminates)

class DDQN_torch:
    def __init__(self, state_size, action_size, device, gamma = 0.99, max_buffer_size = 1e5,
                 batch_size = 64, lr = 1e-3, copy_param = 4, double = False, constraint_func = None,
                 epsilon_init = 1.0, epsilon_decay = 0.99, decay_epsilon_experience = True
                 ):
        """
        DDQN Agent

        Parameters
        ----------
        state_size (int): Dimension of state
        action_size (int): Dimension of action
        device (string): device (cuda:# or cpu)
        gamma (float): discount of future rewards
        max_buffer_size (int): maximum size of internal memory
        batch_size (int)
        lr (float): learning rate
        copy_param (int): How often to copy the train network to target network
        double (bool): Whether to use double DQN or normal DQN
        constraint_func (function): Constraint
        epsilon_init (float): Beginning value of epsilon for epsilon-greedy exploration
        epsilon_decay (float): Rate to decay epsilon
        decay_epsilon_experiences (bool): if True should decay epsilon after every experience, if False, decay epsilon after every episode
        
        """
        self.state_size = state_size
        self.action_size = action_size
        # Initialize Q and Fixed Q networks
        self.train_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.train_network.parameters(), lr=lr)
        # Initilize memory
        self.replay_buffer = ReplayBuffer(max_buffer_size, device)

        #Values
        self.batch_size = batch_size
        self.copy_param = copy_param
        self.gamma = gamma
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.decay_epsilon_experience = decay_epsilon_experience
        self.device = device
        self.double = double

        self.timestep = 0

        self.constraint_func = constraint_func
        # self.curr_const = None
        # self.removed_models = None

    def get_available_actions(self, state):
        if self.constraint_func is not None:
            available_actions = self.constraint_func(state)
            not_available_actions = [action for action in range(self.action_size) if action not in available_actions]
        else:
            available_actions = list(range(self.action_size))
            not_available_actions = []
        return available_actions, not_available_actions

    def add_experience(self, state, action, reward, next_state, terminate, decay = True):
        """
        Add experience to buffer

        Parameters
        ----------
        state (array_like)
        action (int)
        reward (float)
        next_state (array_like)
        terminate (bool)
        """
        self.replay_buffer.add_experience(state, action, reward, next_state, terminate)
        if self.decay_epsilon_experience and decay:
            self.epsilon = max(self.epsilon * self.epsilon_decay, 0.01)
        self.timestep += 1

    def sample_batch(self):
        if self.replay_buffer.n_items < self.batch_size:
            return None
        else:
            return self.replay_buffer.sample_batch(self.batch_size)

    def train(self):
        """
        Updates the train network
        """
        experiences = self.sample_batch()
        if experiences is None:
            return 0
        states, actions, rewards, next_states, terminates = experiences
        # Get the action with max Q value
        action_values_eval = self.target_network(next_states).detach()
        if self.double:
            action_values = self.train_network(next_states).detach()
        else:
            action_values = action_values_eval
        
        if self.constraint_func is not None:
            x = []
            y = []

            for i, ns in enumerate(next_states):
                available_actions, not_available_actions = self.get_available_actions(ns)

                x.extend([i] * len(not_available_actions))
                y.extend(not_available_actions)
            not_available_idxs = (x, y)
            action_values[not_available_idxs] = MINVALUE
        max_next_actions = action_values.argmax(1)
        max_action_values = action_values_eval[range(self.batch_size), max_next_actions].unsqueeze(1)

        Q_target = rewards + (self.gamma * max_action_values * (1 - terminates))
        Q_expected = self.train_network(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.timestep % self.copy_param == 0:
            self.copy_weights()
        return float(loss.detach().cpu().data)

    def add_experience_train(self, state, action, reward, next_state, terminate, decay = True):
        """
        Adds experience and train
        """
        self.add_experience(state, action, reward, next_state, terminate, decay = True)
        return self.train()

    def copy_weights(self):
        """
        Updates the target network with the train network
        """
        for source_parameters, target_parameters in zip(self.train_network.parameters(), self.target_network.parameters()):
            target_parameters.data.copy_(TAU * source_parameters.data + (1.0 - TAU) * target_parameters.data)

    def get_action(self, state, epsilon = None):
        """
        Choose the action

        Parameters
        ----------
        state (array_like): current state of environment
        epsilon (None or float): controls epsilon for exploration, set to 0 if evaluating and set to None to train
        """
        if epsilon is None:
            epsilon = self.epsilon
        rnd = random.random()
        available_actions, not_available_actions = self.get_available_actions(state)

        if rnd < epsilon:
            return np.random.choice(available_actions)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            # set the network into evaluation mode
            self.train_network.eval()
            with torch.no_grad():
                action_values = self.train_network(state)
            # Back to training mode
            self.train_network.train()
            av = action_values.cpu().data.numpy().squeeze()

            if self.constraint_func is not None:
                av[not_available_actions] = MINVALUE
            action = np.argmax(av)
            return action

    def save_model(self, filename):
        '''
        Save the model
        '''
        torch.save(self.train_network.state_dict(), filename + '_train.pt')
        torch.save(self.target_network.state_dict(), filename + '_target.pt')

    def load(self, filename):
        '''
        Load the model
        '''
        self.train_network.load_state_dict(torch.load(filename + '_train.pt'))
        self.target_network.load_state_dict(torch.load(filename + '_target.pt'))
        self.train_network.eval()
        self.target_network.eval()

    def end_episode(self, *args, **kwargs):
        return None


class DQNLoadedPolicy:
    def __init__(self, state_size, action_size, filename, device):
        """
        A simplified DQN agent which executes a saved policy 

        Parameters
        ----------
        state_size (int): Dimension of state
        action_size (int): Dimension of action
        device (string): device (cuda:# or cpu)
        filename (string): filename of saved policy save file
        """
        self.state_size = state_size
        self.action_size = action_size
        self.train_network = QNetwork(state_size, action_size).to(device)
        self.target_network = QNetwork(state_size, action_size).to(device)
        self.load(filename, device)
        self.device = device

    def get_actions(self, state, n_a=1):
        """
        Choose the action

        Parameters
        ----------
        state (array_like): current state of environment
        n_a (int): number of top actions to return
        """
        try:
            state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        except:
            state = state.float().unsqueeze(0).to(self.device)
        self.train_network.eval()
        with torch.no_grad():
            action_values = self.train_network(state)
        av = action_values.cpu().data.numpy().squeeze()
        ai = av.argsort()
        return ai[-n_a:]

    def load(self, filename, device):
        '''
        Load the model
        '''
        self.train_network.load_state_dict(torch.load(filename + '_train.pt', map_location=device))
        self.target_network.load_state_dict(torch.load(filename + '_target.pt', map_location=device))
        self.train_network.eval()
        self.target_network.eval()


