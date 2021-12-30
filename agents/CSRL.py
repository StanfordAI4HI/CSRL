import numpy as np

class CSRL(object):

    def __init__(self, BaseAgent, agent_params, constraint_func_list=[None], 
                 constraint_conf_mult=1, epsilon_decay_list=None, max_value=600, offset=0.5,
                 less_constrained_constraints_list=[[]], more_constrained_constraints_list=[[]],
                 policy_track_length=10, loss_threshold=10, should_eliminate = True):
        """
        CSRL Agent

        Parameters
        ----------
        BaseAgent (class): Base learning agent (ex DDQN_py, rainbow)
        agent_params (dict): Dictionary containing the arguments, with the exceptioon of constraint_funct of the base agent
        constraint_func_list (list): List of constraint functions to sample between
        constraint_conf_mult (float): parameter for multiplying the confidence bound (c)
        epsilon_decay_list (list): List of floats specifying how fast to decay epsilon for epsilon greedy exploration (not needed for the rainbow agent)
        max_value (float): A parameter for scaling the rewards to range of 0 to 1, should be the estimated maximum range of rewards (estimated_max_reward - estimated_min_reward)
        offset (float): A parameter for shifting the rewards to range of 0 to 1, should be estimated_min_reward/max_value, ex. if we expect rewards to be between -300 to +300, the max_value = 600 and offset of 0.5 will shift rewards to between 0 and 1
        
        Constraint Structure parameters:
        -----------------------
        As a running example, consider two constraints, [None (unconstrained), C1]
        less_constrained_constraints_list (list): list of lists that gives the indexes of less constrained constraints for every constraint, for the example it would be [[], [0]] because nothing is less constrained than unconstrained while unconstrained is less constrained that C1
        more_constrained_constraints_list (list): list of lists that gives indexes of more constrained constraints, for the example [[C1], []].
        policy_track_length: Minimum time to wait before elimination
        loss_threshold (float between 0 and 1): Loss limit for elimination (is scaled by max_value)
        should_eliminate (bool): Toggles elimination
        """
        
        self.constraint_func_list = constraint_func_list
        self.max_value = max_value
        self.offset = offset
        self.constraint_conf_mult = constraint_conf_mult

        self.n_constraints = len(constraint_func_list)
        self.N_const_total = {i: 0 for i in range(self.n_constraints)}
        self.N_const_values = {i: [] for i in range(self.n_constraints)}
        self.models = []

        for i, constraint_func in enumerate(constraint_func_list):
            if epsilon_decay_list is not None:
                epsilon_decay = epsilon_decay_list[i]
                self.models.append(
                    BaseAgent(constraint_func=constraint_func, epsilon_decay=epsilon_decay, **agent_params))
            else:
                self.models.append(
                    BaseAgent(constraint_func=constraint_func, **agent_params))
        self.n_user = 0
        self.n_user_total = 0
        self.curr_const = np.random.choice(self.n_constraints)
        self.model_losses = {i: [] for i in range(self.n_constraints)}
        self.consistent = {i: True for i in range(self.n_constraints)}

        self.less_constrained_constraints_list = less_constrained_constraints_list
        self.more_constrained_constraints_list = more_constrained_constraints_list
        self.removed_models = []
        self.policy_track_length = policy_track_length
        self.loss_low = {i: 0 for i in range(self.n_constraints)}
        self.loss_threshold = loss_threshold * max_value
        self.ep_length = 0
        self.debugging = []
        self.should_eliminate = should_eliminate
        self.not_updated_models = list(range(self.n_constraints))

    def conf_const_calc(self, const_idx):
        '''
        Calculate UCB confidence bound
        '''
        return self.constraint_conf_mult * np.sqrt(2 * np.log(self.n_user) / self.N_const_total[const_idx])

    def new_constraint(self):
        '''
        Eliminates previous constraint if necessary and chooses new constraint
        '''
        self.n_user_total += 1
        removed_constraint = None
        # Take every constraint at least once:
        if len(self.not_updated_models) > 0:
            self.curr_const = np.random.choice(self.not_updated_models)
        else:
            est_const_means = [np.mean(self.N_const_values[const_idx]) / self.max_value + self.offset
                               for const_idx in range(self.n_constraints)]
            est_const_means_samples = [np.mean(self.N_const_values[const_idx][-self.policy_track_length:]) / self.max_value + self.offset
                                       if const_idx not in self.removed_models and len(self.N_const_values[const_idx]) > self.policy_track_length
                                       else -1000
                                       for const_idx in range(self.n_constraints)]


            const_idx = self.curr_const
            if self.should_eliminate:
                max_value,  max_idx, policy_mean = np.max(est_const_means_samples), np.argmax(est_const_means_samples), est_const_means_samples[const_idx]

                if const_idx != max_idx and \
                        len(self.N_const_values[const_idx]) > self.policy_track_length and \
                        self.loss_low[const_idx] > self.policy_track_length and \
                        len(self.less_constrained_constraints_list[const_idx]) > 0 and \
                        const_idx not in self.less_constrained_constraints_list[max_idx] and \
                        len([cidx for cidx in self.more_constrained_constraints_list[const_idx] if
                             cidx not in self.removed_models]) == 0:
                    self.debugging.append((const_idx, policy_mean, max_idx, max_value, self.n_user, self.loss_low))
                    print(f"removing model: {const_idx}, max model:{max_idx}, {np.mean(self.N_const_values[const_idx]):.2f}, {np.mean(self.N_const_values[max_idx]):.2f}, {est_const_means_samples[const_idx]:.2f}, {est_const_means_samples[max_idx]:.2f}")
                    self.removed_models.append(const_idx)
                    removed_constraint = const_idx

            est_const_values = [est_const_means[const_idx] + self.conf_const_calc(const_idx)
                                if const_idx not in self.removed_models else -1000
                                for const_idx in range(self.n_constraints)]

            self.curr_const = np.argmax(est_const_values)
        return removed_constraint

    def end_episode(self, episode_reward):
        '''
        Processes episode information when episode ends
        
        Parameters
        ----------
        episode_reward: The reward for the most recent episode
        '''
        if self.curr_const in self.not_updated_models:
            self.not_updated_models.remove(self.curr_const)
        self.n_user += 1
        self.N_const_values[self.curr_const].append(episode_reward)
        self.N_const_total[self.curr_const] += 1

        if np.mean(self.model_losses[self.curr_const][-self.ep_length:]) < self.loss_threshold:
            self.loss_low[self.curr_const] += 1
        else:
            self.loss_low[self.curr_const] = 0
        self.consistent = {i: True for i in range(self.n_constraints)}
        self.ep_length = 0
        return self.new_constraint()

    def get_action(self, state, epsilon=None):
        """
        Choose the action

        Parameters
        ----------
        state (array_like): current state of environment
        """
        return self.models[self.curr_const].get_action(state, epsilon)

    def add_experience_train(self, state, action, reward, next_state, terminate):
        '''
        Adds the transition experience and trains models
        
        Note for speed, experiences are only added to less constrained constraints of the current constraint (as opposed to checking if the transition satisfies the constraint for every constraint)
        '''
        total_loss = 0.0
        self.ep_length += 1
        for model_idx in self.less_constrained_constraints_list[self.curr_const] + [self.curr_const]:
            if model_idx not in self.removed_models:
                model = self.models[model_idx]
                loss = model.add_experience_train(state, action, reward, next_state, terminate, decay=(model_idx == self.curr_const))
                if loss is not None:
                    self.model_losses[model_idx].append(loss)
                    total_loss += loss
        return total_loss