
import numpy as np
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Dense ,InputLayer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
MINVALUE= -100000000

class QNetwork(tf.keras.Model):
    
    def __init__(self, state_size, action_size, hidden_units):
        """
        Network for DDQN Agent

        Parameters
        ----------
        action_size (int): number of actions
        hidden_units (array like): an array of hidden units for the QNetwork
        """
        super(QNetwork, self).__init__()
        self.input_layer = InputLayer(input_shape=(state_size,))
        self.hidden_layers = []
        for n_units in hidden_units:
            self.hidden_layers.append(Dense(
                n_units, activation='tanh', kernel_initializer='RandomNormal'))
        self.output_layer = Dense(
            action_size, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, inputs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

class ReplayBuffer(object):
    def __init__(self, max_buffer_size, state_size):
        """
        Experience Replay Buffer

        Parameters
        ----------
        state_size (int): continuous state dimenion
        max_buffer_size (int): maximum size of internal memory
        
        """
        self.max_buffer_size = max_buffer_size
        self.state_size = state_size
        self.n_items = 0
        self.curr_idx = 0
        self.actions = np.empty(self.max_buffer_size, dtype = np.int32)
        self.rewards = np.empty(self.max_buffer_size, dtype = np.float32)
        self.states = np.empty((self.max_buffer_size, self.state_size), dtype = np.float32)
        self.next_states = np.empty((self.max_buffer_size, self.state_size), dtype = np.float32)
        self.terminates = np.empty(self.max_buffer_size, dtype = bool)
    
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
        self.actions[self.curr_idx] = action
        self.states[self.curr_idx] = state
        self.rewards[self.curr_idx] = reward
        self.next_states[self.curr_idx] = next_state
        self.terminates[self.curr_idx] = terminate
        
        self.curr_idx = (self.curr_idx + 1) % self.max_buffer_size
        self.n_items = min(self.n_items + 1, self.max_buffer_size)
    
    def sample_batch(self, batch_size):
        idxs = np.random.randint(low = 0, high = self.n_items, size = batch_size)
        return (self.states[idxs], self.actions[idxs], self.rewards[idxs], self.next_states[idxs], self.terminates[idxs])

class DDQN_tf(object):
    def __init__(self, state_size, action_size, hidden_units, gamma, max_buffer_size,
                 batch_size, lr, copy_param, double = False, constraint_func = None,
                 epsilon_init = 1, epsilon_decay = 0.995, decay_epsilon_experience = True):
        """
        DDQN Agent

        Parameters
        ----------
        state_size (int): Dimension of state
        action_size (int): Dimension of action
        hidden_units (array like): an array of hidden units for the QNetwork
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
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.copy_param = copy_param
        self.double = double
        
        self.constraint_func = constraint_func
        
        self.optimizer = tf.optimizers.Adam(lr)
        self.train_network = QNetwork(state_size, action_size, hidden_units)
        self.target_network = QNetwork(state_size, action_size, hidden_units)
        
        self.replay_buffer = ReplayBuffer(max_buffer_size, state_size)
        self.max_buffer_size = max_buffer_size
        self.epsilon = epsilon_init
        self.epsilon_decay = epsilon_decay
        self.decay_epsilon_experience = decay_epsilon_experience
        
        self.updated_once = False
        self.updated_last_episode = False
        self.timestep = 0
    
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
            self.epsilon = self.epsilon * self.epsilon_decay
        self.timestep += 1
        
    def _predict(self, inputs):
        return self.train_network(np.atleast_2d(inputs.astype('float32')))
    
    def sample_batch(self):
        if self.replay_buffer.n_items < self.batch_size:
            return None
        else :
            return self.replay_buffer.sample_batch(self.batch_size)
        
    def train(self):
        """
        Updates the train network
        """
        return_tuple = self.sample_batch()
        if return_tuple is None:
            return 0
        self.updated_once = True
        states, actions, rewards, states_next, dones = return_tuple
        
        if self.double:
            predicted_values = self.train_network.predict(states_next)
        else:
            predicted_values = self.target_network.predict(states_next)
        
        if self.constraint_func is not None:
            x = []
            y = []
        
            for i, ns in enumerate(states_next):
                available_actions, not_available_actions = self.get_available_actions(ns)
                
                x.extend([i]*len(not_available_actions))
                y.extend(not_available_actions)
            not_available_idxs = (x, y)
            predicted_values[not_available_idxs] = MINVALUE
        
        if self.double:
            max_actions = predicted_values.argmax(axis = 1)
            next_action_q = self.target_network.predict(states_next)
            value_next = next_action_q[range(self.batch_size), max_actions]
        else:
            value_next = np.max(predicted_values, axis=1)
        
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self._predict(states) * tf.one_hot(actions, self.action_size), axis=1)
            loss = tf.math.reduce_mean(tf.square(actual_values - selected_action_values))
        variables = self.train_network.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        if self.timestep % self.copy_param == 0:
            self.copy_weights()
        
        return loss.numpy()

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
        available_actions, not_available_actions = self.get_available_actions(state)
        predicted_actions_values = self._predict(np.atleast_2d(state)).numpy()[0]
        predicted_actions_values[not_available_actions] = MINVALUE
        action = np.argmax(predicted_actions_values)

        if np.random.random() < epsilon:
            action =  np.random.choice(available_actions)
        return action
    
    
    
    def add_experience_train(self, state, action, reward, next_state, terminate, decay = True):
        """
        Adds experience and train
        """
        self.add_experience(state, action, reward, next_state, terminate, decay = decay)
        return self.train()

    def copy_weights(self):
        """
        Updates the target network with the train network
        """
        target_variables = self.target_network.trainable_variables
        train_variables = self.train_network.trainable_variables
        for target_v, train_v in zip(target_variables , train_variables):
            target_v.assign(train_v.numpy())
    
    def check_state_action(self, state, action):
        return action in self.constraint_func(state)
    
    def save_model(self, save_path, save_path_target =None):
        self.train_network.save(save_path)
        if save_path_target is not None:
            self.target_network.save(save_path_target)
    
    def end_episode(self, *args, **kwargs):
        if not self.decay_epsilon_experience:
            self.epsilon = self.epsilon * self.epsilon_decay
        return None

class DQNLoadedPolicy(object):
    def __init__(self, action_size, filename):
        """
        A simplified DQN agent which executes a saved policy 

        Parameters
        ----------
        action_size (int): Dimension of action
        filename (string): filename of saved policy save file
        """
        self.action_size = action_size
        self.train_network = tf.keras.models.load_model(filename)
    
    def _predict(self, inputs):
        return self.train_network(np.atleast_2d(inputs.astype('float32')))
    
    def get_action_list(self, states, *args):
        predicted_actions_values = self._predict(np.atleast_2d(states)).numpy()[0]
        actions = list(range(self.action_size))
        
        sorted_actions = [x for _,x in sorted(zip(predicted_actions_values,actions), key=lambda pair: -pair[0])]
        
        best_action = np.argmax(predicted_actions_values)
        return best_action, sorted_actions
   
    def get_action(self, states, *args):
        return self.get_action_list(states)[0]