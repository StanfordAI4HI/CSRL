#Adapted from Rainbow tutorial of https://github.com/Curt-Park/rainbow-is-all-you-need

import math
import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
from copy import deepcopy
from agents.segment_tree import MinSegmentTree, SumSegmentTree
MINVALUE = -1000

class ReplayBuffer:

    def __init__(self, state_size, max_buffer_size, batch_size = 32, n_step = 1, gamma = 0.99):
        self.states = np.zeros([max_buffer_size, state_size], dtype=np.float32)
        self.next_states = np.zeros([max_buffer_size, state_size], dtype=np.float32)
        self.actions = np.zeros([max_buffer_size], dtype=np.float32)
        self.rewards = np.zeros([max_buffer_size], dtype=np.float32)
        self.terminates = np.zeros(max_buffer_size, dtype=np.float32)
        self.max_buffer_size, self.batch_size = max_buffer_size, batch_size
        self.ptr, self.size, = 0, 0

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def add_experience(self, state, action, reward, next_state, terminate):
        transition = (state, action, reward, next_state, terminate)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make a n-step transition
        reward, next_state, terminate = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        state, action = self.n_step_buffer[0][:2]

        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.terminates[self.ptr] = terminate
        self.ptr = (self.ptr + 1) % self.max_buffer_size
        self.size = min(self.size + 1, self.max_buffer_size)

        return self.n_step_buffer[0]

    def sample_batch(self):
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            states=self.states[idxs],
            next_states=self.next_states[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            terminates=self.terminates[idxs],
            # for N-step Learning
            indices=idxs,
        )

    def sample_batch_from_idxs(self, idxs):
        # for N-step Learning
        return dict(
            states=self.states[idxs],
            next_states=self.next_states[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            terminates=self.terminates[idxs],
        )

    def _get_n_step_info(self, n_step_buffer, gamma):
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        reward, next_state, terminate = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            reward = r + gamma * reward * (1 - d)
            next_state, terminate = (n_o, d) if d else (next_state, terminate)

        return reward, next_state, terminate

    def __len__(self):
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(self, state_size, max_buffer_size, batch_size = 32, alpha = 0.6, n_step = 1, gamma = 0.99):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(
            state_size, max_buffer_size, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_buffer_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def add_experience(self, state, action, reward, next_state, terminate):
        """Store experience and priority."""
        transition = super().add_experience(state, action, reward, next_state, terminate)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_buffer_size

        return transition

    def sample_batch(self, beta = 0.4):
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        states = self.states[indices]
        next_states = self.next_states[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        terminates = self.terminates[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            states=states,
            next_states=next_states,
            actions=actions,
            rewards=rewards,
            terminates=terminates,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self):
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx, beta):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.



    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(
            self, in_features, out_features, std_init = 0.75):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size):
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


class QNetwork(nn.Module):
    def __init__(
            self, in_dim, out_dim, atom_size, support, std_init = 0.75, hidden_units = [128, 128]):
        """
        Network for Rainbow Agent

        Parameters
        ----------
        action_size (int): number of actions
        hidden_units (array like): an array of hidden units for the QNetwork
        """
        super(QNetwork, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, hidden_units[0]),
            nn.ReLU(),
        )

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(hidden_units[0], hidden_units[1])
        self.advantage_layer = NoisyLinear(hidden_units[1], out_dim * atom_size, std_init=std_init)

        # set value layer
        self.value_hidden_layer = NoisyLinear(hidden_units[0], hidden_units[1])
        self.value_layer = NoisyLinear(hidden_units[1], atom_size, std_init=std_init)

    def forward(self, x):
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x):
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


class Rainbow:

    def __init__(
            self,
            state_size,
            action_size,
            device = None,
            gamma=0.99,
            #MEMORY
            max_buffer_size=int(1e5),
            batch_size=64,
            lr=1e-4,
            copy_param=4,
            constraint_func=None,
            # PER parameters
            alpha = 0.2,
            beta = 0.6,
            max_beta = 1.0,
            prior_eps = 1e-6,
            # Categorical DQN parameters
            v_min = 0.0,
            v_max = 200.0,
            atom_size = 51,
            # N-step Learning
            n_step = 1,
            total_frames = 20000,
            std_init = 0.75,
            *args, **kwargs
    ):
        """Initialization.

        Args:
            state_size (int): continuous state dimenion
            action_size (int): number of actions
            device (string): device (cuda:# or cpu)
            gamma (float): discount of future rewards
            max_buffer_size (int): maximum size of internal memory
            batch_size (int)
            lr (float): learning rate
            copy_param (int): How often to copy the train network to target network
            constraint_func (function): Constraint
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
        """
        self.action_size = action_size
        self.batch_size = batch_size
        self.copy_param = copy_param
        self.gamma = gamma
        # NoisyNet: All attributes related to epsilon are removed

        # device: cpu / gpu
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = device
        #         print(self.device)

        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.max_beta = max_beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            state_size, max_buffer_size, batch_size, alpha=alpha
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                state_size, max_buffer_size, batch_size, n_step=n_step, gamma=gamma
            )

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.train_network = QNetwork(
            state_size, action_size, self.atom_size, self.support, std_init=std_init
        ).to(self.device)
        self.target_network = QNetwork(
            state_size, action_size, self.atom_size, self.support, std_init=std_init
        ).to(self.device)
        self.target_network.load_state_dict(self.train_network.state_dict())
        self.target_network.eval()

        # optimizer
        self.optimizer = optim.Adam(self.train_network.parameters(), lr=lr)

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

        self.frame_idx = 0
        self.total_frames = total_frames
        self.update_cnt = 0
        self.constraint_func = constraint_func
        self.removed_models = None

        self.timestep = 0

    def get_available_actions(self, state):
        if self.constraint_func is not None:
            available_actions = self.constraint_func(state)
            not_available_actions = [action for action in range(self.action_size) if action not in available_actions]
        else:
            available_actions = list(range(self.action_size))
            not_available_actions = []
        return available_actions, not_available_actions

    def get_action(self, state, *args, **kwargs):
        """
        Choose the action

        Parameters
        ----------
        state (array_like): current state of environment
        """
        # NoisyNet: no epsilon greedy action selection
        available_actions, not_available_actions = self.get_available_actions(state)

        action_values = self.train_network(
            torch.FloatTensor(state).to(self.device)
        )
        av = action_values.detach().cpu().data.numpy().squeeze()
        if self.constraint_func is not None:
            av[not_available_actions] = MINVALUE
        selected_action = np.argmax(av)

        return selected_action

    def update_model(self):
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.train_network.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.train_network.reset_noise()
        self.target_network.reset_noise()

        return loss.item()

    def update_transitions(self, transitions):
        losses = []
        for transition in transitions:
            #         for frame_idx in range(1, num_frames + 1):
            self.frame_idx += 1
            self.store_transition(transition)

            # NoisyNet: removed decrease of epsilon

            # PER: increase beta
            fraction = min(self.frame_idx / self.total_frames, 1.0)
            self.beta = self.beta + fraction * (self.max_beta - self.beta)

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                self.update_cnt += 1

                # if hard update is needed
                if self.update_cnt % self.target_update == 0:
                    self.copy_weights()

        #         self.env.close()

        return np.sum(losses), np.mean(losses)

    def add_experience(self, state, action, reward, next_state, terminate):
        transition = (state, action, reward, next_state, terminate)
        # N-step transition
        if self.use_n_step:
            one_step_transition = self.memory_n.add_experience(*transition)
        # 1-step transition
        else:
            one_step_transition = transition

        # add a single step transition
        if one_step_transition:
            self.memory.add_experience(*one_step_transition)
        self.frame_idx += 1
        self.timestep += 1

        # PER: increase beta
        fraction = min(self.frame_idx / self.total_frames, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)

    def train(self):
        # if training is ready
        total_loss = 0
        if len(self.memory) >= self.batch_size:
            loss = self.update_model()
            total_loss += loss
            self.update_cnt += 1

            # if hard update is needed
            if self.update_cnt % self.copy_param == 0:
                self.copy_weights()
        return total_loss

    def end_episode(self, *args, **kwargs):
        pass

    def add_experience_train(self, state, action, reward, next_state, terminate, decay = True):
        if decay:
            self.add_experience(state, action, reward, next_state, terminate)
            return self.train()
        else:
            return None

    def _compute_dqn_loss(self, samples, gamma):
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["states"]).to(device)
        next_state = torch.FloatTensor(samples["next_states"]).to(device)
        action = torch.LongTensor(samples["actions"]).to(device)
        reward = torch.FloatTensor(samples["rewards"].reshape(-1, 1)).to(device)
        terminate = torch.FloatTensor(samples["terminates"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            #             next_action = self.dqn(next_state).argmax(1)

            next_action_values = self.train_network(next_state).detach()
            if self.constraint_func is not None:
                x = []
                y = []

                for i, ns in enumerate(next_state):
                    available_actions, not_available_actions = self.get_available_actions(ns)

                    x.extend([i] * len(not_available_actions))
                    y.extend(not_available_actions)
                not_available_idxs = (x, y)
                next_action_values[not_available_idxs] = MINVALUE
            next_action = next_action_values.argmax(1)
            #             print(next_action.shape)
            next_dist = self.target_network.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - terminate) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                    .unsqueeze(1)
                    .expand(self.batch_size, self.atom_size)
                    .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.train_network.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def copy_weights(self):
        """Hard update: target <- local."""
        self.target_network.load_state_dict(self.train_network.state_dict())