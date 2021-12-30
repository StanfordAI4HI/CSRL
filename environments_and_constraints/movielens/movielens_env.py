import re
from itertools import product

import numpy as np

def load_environment(genre_params_used, dropout_type = 1):
    actions_list = None

    #original parameters from Warlop et. al
    if genre_params_used == 0:
        d = 5
        d2 = 0
        d3 = 0
        genre_params_all = np.array([
            [3.1, 0.54, -1.08, 0.78, -0.22, 0.02],
            [3.34, 0.54, -1.08, 0.78, -0.22, 0.02],
            [3.51, 0.86, -2.7, 3.06, -1.46, 0.24],
            [3.4, 1.26, -2.9, 2.76, -1.14, 0.16],
            [2.75, 1.0, 0.94, -1.86, 0.94, -0.16],
            [3.52, 0.1, 0.0, -0.3, 0.2, -0.04],
            [3.37, 0.32, 1.12, -3.0, 2.26, -0.54],
            [3.54, -0.68, 1.84, -2.04, 0.82, -0.12],
            [3.3, 0.64, -1.32, 1.1, -0.38, 0.02],
            [3.4, 1.38, -3.44, 3.62, -1.62, 0.24]
        ])
        
    #w = 3
    if genre_params_used == 3:
        d = 5
        d2 = 2
        d3 = 2
        genre_params_all = np.array([[ 3.37629188e+00, -1.12659451e+00,  2.80606620e+00,
                                     -3.40688115e+00,  1.80481753e+00, -3.46558986e-01,
                                      5.13155350e-01, -2.48608719e-01,  1.58030328e-01,
                                      2.97825267e-02],
                                    [ 3.34305623e+00, -8.07533528e-01,  2.19467693e+00,
                                     -2.95525803e+00,  1.65674777e+00, -3.29694591e-01,
                                      4.25951219e-01, -2.47352870e-01,  8.46376654e-02,
                                      7.38255108e-02],
                                    [ 2.92426896e+00, -1.05963582e+00,  2.70303335e+00,
                                     -2.99818949e+00,  1.37876319e+00, -2.15330501e-01,
                                      1.39518117e+00, -6.12065965e-01,  1.39422712e-01,
                                     -3.32428870e-02],
                                    [ 3.07230683e+00,  6.31420621e-03, -1.06741780e+00,
                                      1.83106381e+00, -1.15263604e+00,  2.43853353e-01,
                                      9.99452046e-01, -4.42089808e-01,  5.48413518e-02,
                                      9.99894890e-03],
                                    [ 2.87315396e+00,  1.50778933e+00, -2.22763362e+00,
                                      1.74576672e+00, -6.15789828e-01,  7.42798229e-02,
                                      1.01434652e+00, -4.44499044e-01, -1.80627389e-01,
                                      4.81254405e-03],
                                    [ 2.62936523e+00,  4.79284944e-02,  7.19272350e-01,
                                     -3.48280769e-01, -4.92690896e-02,  3.87238157e-02,
                                      1.12201961e+00, -3.07386430e-01, -2.79158303e-01,
                                     -2.65497722e-01],
                                    [ 2.34087117e+00,  3.18550408e+00, -1.06011158e+01,
                                      1.49981670e+01, -8.78542995e+00,  1.79120605e+00,
                                      2.53015294e+00, -1.31622045e+00, -2.86115217e-01,
                                     -1.71427070e-01],
                                    [ 2.73899583e+00,  1.75495737e+00, -7.28826516e+00,
                                      1.01180534e+01, -5.64550573e+00,  1.07978386e+00,
                                      1.27894489e+00, -6.59868728e-01, -5.81185288e-02,
                                     -2.33236793e-02],
                                    [ 2.55126515e+00,  1.64327899e+00, -6.37497923e+00,
                                      8.25899046e+00, -4.36110407e+00,  8.14791782e-01,
                                      2.18369781e+00, -1.21640371e+00,  3.19605917e-02,
                                      6.59123351e-02],
                                    [ 3.55198030e+00, -6.83466784e-01,  6.29244300e-01,
                                      8.69119639e-01, -1.18625421e+00,  3.30614369e-01,
                                      8.94250852e-02,  2.35538634e-01,  9.68816645e-02,
                                     -1.85764820e-01]])
        idx_used = [0, 1, 2, 3, 4, 5, 7]
        more_constrained_constraints = {
            'g1': ['g2', 'g3', 'g4', 'e1', 'e2', 'e3', 'o1', 'o2', 'o3', 'o4', 't1', 't2', 't3'],
            'g2': ['g3', 'g4', 'e2', 'e3', 'o2', 'o3', 'o4', 't2', 't3'],
            'g3': ['g4', 'e3',  'o3', 'o4', 't3'],
            'g4': ['o4'],
            'e1': ['o1', 't1'],
            'e2': ['o2', 't2'],
            'e3': ['o3', 't3'],
            'o1': ['t1'],
            'o2': ['t2'],
            'o3': ['t3'],
            'o4': [],
            't1': [],
            't2': [],
            't3': []
        }
        
    #7: w = 4
    if genre_params_used == 4:
        d = 5
        d2 = 2
        d3 = 2
        genre_params_all = np.array([[ 3.37629188e+00, -1.12659451e+00,  2.80606620e+00,
                                         -3.40688115e+00,  1.80481753e+00, -3.46558986e-01,
                                          5.13155350e-01, -2.48608719e-01,  1.58030328e-01,
                                          2.97825267e-02],
                                        [ 3.34305623e+00, -8.07533528e-01,  2.19467693e+00,
                                         -2.95525803e+00,  1.65674777e+00, -3.29694591e-01,
                                          4.25951219e-01, -2.47352870e-01,  8.46376654e-02,
                                          7.38255108e-02],
                                        [ 2.92426896e+00, -1.05963582e+00,  2.70303335e+00,
                                         -2.99818949e+00,  1.37876319e+00, -2.15330501e-01,
                                          1.39518117e+00, -6.12065965e-01,  1.39422712e-01,
                                         -3.32428870e-02],
                                        [ 3.07230683e+00,  6.31420621e-03, -1.06741780e+00,
                                          1.83106381e+00, -1.15263604e+00,  2.43853353e-01,
                                          9.99452046e-01, -4.42089808e-01,  5.48413518e-02,
                                          9.99894890e-03],
                                        [ 2.87315396e+00,  1.50778933e+00, -2.22763362e+00,
                                          1.74576672e+00, -6.15789828e-01,  7.42798229e-02,
                                          1.01434652e+00, -4.44499044e-01, -1.80627389e-01,
                                          4.81254405e-03],
                                        [ 2.62936523e+00,  4.79284944e-02,  7.19272350e-01,
                                         -3.48280769e-01, -4.92690896e-02,  3.87238157e-02,
                                          1.12201961e+00, -3.07386430e-01, -2.79158303e-01,
                                         -2.65497722e-01],
                                        [ 2.34087117e+00,  3.18550408e+00, -1.06011158e+01,
                                          1.49981670e+01, -8.78542995e+00,  1.79120605e+00,
                                          2.53015294e+00, -1.31622045e+00, -2.86115217e-01,
                                         -1.71427070e-01],
                                        [ 2.73899583e+00,  1.75495737e+00, -7.28826516e+00,
                                          1.01180534e+01, -5.64550573e+00,  1.07978386e+00,
                                          1.27894489e+00, -6.59868728e-01, -5.81185288e-02,
                                         -2.33236793e-02],
                                        [ 2.55126515e+00,  1.64327899e+00, -6.37497923e+00,
                                          8.25899046e+00, -4.36110407e+00,  8.14791782e-01,
                                          2.18369781e+00, -1.21640371e+00,  3.19605917e-02,
                                          6.59123351e-02],
                                        [ 3.55198030e+00, -6.83466784e-01,  6.29244300e-01,
                                          8.69119639e-01, -1.18625421e+00,  3.30614369e-01,
                                          8.94250852e-02,  2.35538634e-01,  9.68816645e-02,
                                         -1.85764820e-01]])
        idx_used = [0, 1, 2, 3, 4, 5, 7]
        more_constrained_constraints = {
            'g1': ['g2', 'g3', 'g4', 'g5','e1', 'e2', 'e3','e4', 'o1', 'o2', 'o3', 'o4', 't1', 't2', 't3'],
            'g2': ['g3', 'g4', 'g5', 'e2', 'e3','e4', 'o2', 'o3', 'o4', 't2', 't3'],
            'g3': ['g4', 'g5', 'e3','e4', 'o3', 'o4', 't3'],
            'g4': ['e4', 'o4'],
            'g5': [],
            'e1': ['o1', 't1'],
            'e2': ['o2', 't2'],
            'e3': ['o3', 't3'],
            'e4': ['o4'],
            'o1': ['t1'],
            'o2': ['t2'],
            'o3': ['t3'],
            'o4': [],
            't1': [],
            't2': [],
            't3': []
        }
    
    #If w = 5
    if genre_params_used == 5:
        d = 5
        d2 = 2
        d3 = 2
        genre_params_all = np.array([[ 2.95244643e+00, -2.37562043e-01,  2.73210452e-01, -3.65967087e-01,  2.17245757e-01, -4.64866721e-02,
             1.15951313e+00, -5.53716918e-01,  1.13529539e-01, 3.49404610e-02],
           [ 3.14727744e+00, -1.48929543e-01, -4.66673016e-03, -6.08495497e-02,  8.45456567e-02, -2.74721718e-02,
             7.22251784e-01, -3.40896001e-01, -9.51513723e-03, 6.17076161e-02],
           [ 2.80291132e+00, -8.48142454e-01,  8.47764724e-01, -5.86231540e-01,  1.12048792e-01,  1.54119969e-02,
             1.43609758e+00, -6.74092328e-01,  6.97159632e-01, -1.63815855e-01],
           [ 2.99961275e+00,  1.70482746e-01, -1.32680117e+00, 1.89599831e+00, -1.10270280e+00,  2.20893654e-01,
             1.00042208e+00, -4.04661344e-01,  4.51112930e-02, 4.40304106e-02],
           [ 2.84887487e+00,  8.21225961e-01, -5.55323154e-01, 1.43891086e-01,  6.22016631e-02, -2.93729990e-02,
             1.13967973e+00, -4.32102276e-01, -3.01339435e-01, 4.49675347e-02],
           [ 2.88540813e+00, -2.69372527e-01,  7.86067709e-01, -8.34251776e-01,  4.21578988e-01, -8.02010892e-02,
             8.98321374e-01, -3.57576153e-01,  1.01106237e-01, -2.17931568e-01],
           [ 2.35824570e+00,  2.40495282e+00, -6.58961083e+00, 9.62808313e+00, -5.91597368e+00,  1.24743726e+00,
             2.40253107e+00, -1.09797851e+00, -9.66962828e-01, 2.71241095e-01],
           [ 2.86144577e+00,  8.45771719e-01, -3.06230082e+00, 3.85720705e+00, -1.95082927e+00,  3.32741667e-01,
             9.15401376e-01, -3.64389373e-01, -2.07949768e-01, 6.23454811e-02],
           [ 2.31823510e+00,  5.55985046e-01, -2.09351094e+00, 2.43445886e+00, -1.11662363e+00,  1.81520264e-01,
             2.37920498e+00, -1.10350742e+00,  1.19889704e-01, -5.63706843e-03],
           [ 3.38765953e+00, -4.86814090e-01, -6.27685766e-01, 1.67329702e+00, -1.12284390e+00,  2.32011922e-01,
             4.12224933e-01, -2.24776960e-01,  1.07379871e+00, -6.39825248e-01]
        ])
        idx_used = [0, 1, 2, 3, 4, 5, 7]
        more_constrained_constraints = {
            'g1': ['g2', 'g3', 'g4', 'g5','e1', 'e2', 'e3','e4', 'o1', 'o2', 'o3', 'o4', 't1', 't2', 't3'],
            'g2': ['g3', 'g4', 'g5', 'e2', 'e3','e4', 'o2', 'o3', 'o4', 't2', 't3'],
            'g3': ['g4', 'g5', 'e3','e4', 'o3', 'o4', 't3'],
            'g4': ['e4', 'o4'],
            'g5': [],
            'e1': ['o1', 't1'],
            'e2': ['o2', 't2'],
            'e3': ['o3', 't3'],
            'e4': ['o4'],
            'o1': ['t1'],
            'o2': ['t2'],
            'o3': ['t3'],
            'o4': [],
            't1': [],
            't2': [],
            't3': []
        }

    if genre_params_used in [3, 4, 5]:
        dropout_p_min = 0.01
        Genres_all_all = ['Action', 'Comedy', 'Adventure', 'Thriller', 'Drama', 'Children', 'Crime', 'Horror', 'Sci-Fi', 'Animation']
        
    
        nA = 5
        w = genre_params_used
        
        dropout_p_real_20 = [1, 0.01247401, 0.01296435, 0.01078655, 0.01033947, 0.00937151, 0.00834071]
        dropout_p_var_act_all = np.array([[1, 0.01517486572265625, 0.016082763671875, 0.01363372802734375, 0.01232147216796875, 0.0120697021484375, 0.011016845703125],
                             [1, 0.01390838623046875, 0.0148162841796875, 0.01236724853515625, 0.01105499267578125, 0.01080322265625, 0.0097503662109375],
                             [1, 0.0115814208984375, 0.01248931884765625, 0.010040283203125, 0.00872802734375, 0.00847625732421875, 0.00742340087890625],
                             [1, 0.0128021240234375, 0.01371002197265625, 0.011260986328125, 0.00994873046875, 0.00969696044921875, 0.00864410400390625],
                             [1, 0.0122833251953125, 0.01319122314453125, 0.0107421875,  0.009429931640625, 0.00917816162109375,  0.00812530517578125],
                             [1, 0.015960693359375,  0.01686859130859375, 0.0144195556640625, 0.0131072998046875, 0.01285552978515625, 0.01180267333984375],
                             [1, 0.0130615234375, 0.01396942138671875, 0.0115203857421875, 0.0102081298828125, 0.00995635986328125, 0.00890350341796875],
                             [1, 0.0173492431640625, 0.01825714111328125, 0.01580810546875, 0.014495849609375, 0.01424407958984375, 0.01319122314453125],
                             [1, 0.016937255859375, 0.01784515380859375, 0.0153961181640625, 0.0140838623046875, 0.01383209228515625, 0.01277923583984375],
                             [1, 0.00977325439453125, 0.01068115234375, 0.00823211669921875, 0.00691986083984375, 0.0066680908203125, 0.005615234375]])
        
        dropout_p_var = dropout_p_real_20
        max_life = 500
        max_lav = 1500
        Genres_all = Genres_all_all
        genre_params = genre_params_all
        if idx_used:
            Genres_all = [x for i, x in enumerate(Genres_all_all) if i in idx_used]
            genre_params = genre_params_all[idx_used]
            dropout_p_var_act = dropout_p_var_act_all[idx_used]

        Genres = Genres_all
        if nA:
            Genres = Genres_all[0:nA]
        nA = len(Genres)
        reset_state = str([-1 for i in range(w)])

    
    if genre_params_used == 0:
        dropout_p_min = 0.015
        nA = 10
        genre_params = genre_params_all
        dropout_p_var_act = dropout_p_var_act_all

    actions = range(nA)

    states_ss = list(product(range(nA), repeat=w)) # all possibles states
    additional_states = []
    for i in range(1, w+1):
        state_end = [-1] * i
        state_begins = list(product(range(nA), repeat=w-i))
        for state_begin in state_begins:
            additional_states.append(tuple(list(state_begin) + state_end))
    states = states_ss + additional_states
    states_str = {str(list(state)):list(state) for state in states}
    states = list(states_str.keys())
    state_to_idx = {state:idx for idx, state in enumerate(states)}
    
    nS = len(states)

    state_action_idx_dict = {}
    idx_state_action_dict = {}
    i = 0
    for s_str in states:
        state_action_idx_dict[s_str] = {}
        for a in actions:
            state_action_idx_dict[s_str][a] = i
            idx_state_action_dict[i] = (s_str,a)
            i += 1

    transitions = {}
    state_action_uniques = {}
    state_uniques = {}
    amount_flex = {}
    for state, s in states_str.items():
        transitions[state] = {}
        state_action_uniques[state] = np.zeros(nA)
        s_nn = [s_a for s_a in s if s_a != -1]
        s_n = [s_a for s_a in s if s_a == -1]
        state_uniques[state] = len(set(s_nn))
        amount_flex[state] = len(s_n)
        for a in range(nA):
            total = s_nn + [int(a)]
            state_action_uniques[state][a] = len(set(total))
            ns = s[:-1]
            ns = [int(a)] + ns
            transitions[state][a] = str(ns)
    
    var_x = var_x_generator(w, d2 = d2, d3 =d3, genre_params_used = genre_params_used, actions = actions_list)

    reward = reward_func_generator(x_func = var_x, genre_params = genre_params, d = d, noise_std = 0.1)
    reward_nonoise = reward_func_generator(x_func = var_x, genre_params = genre_params, d = d, noise_std = 0)
    reward_func_generator_constrained = reward_func_generator_constrained_generator(x_func = var_x, d = d, genre_params = genre_params)

    if dropout_type == 0:
        dropout_func_generator = dropout_func_func_generator0(state_action_uniques = state_action_uniques)
        dropout_func = dropout_func_generator(dropout_p_var = dropout_p_var)
        dropout_func_binary = dropout_func_binary_generator(dropout_func= dropout_func)
    if dropout_type == 1:
        dropout_func_generator = dropout_func_func_generator1(state_action_uniques = state_action_uniques)
        dropout_func = dropout_func_generator(dropout_p_var_act = dropout_p_var_act)
        dropout_func_binary = dropout_func_binary_generator(dropout_func= dropout_func)
    if dropout_type == 2:
        dropout_func_generator = dropout_func_func_generator2(state_action_uniques = state_action_uniques)
        dropout_func = dropout_func_generator(dropout_p_min = dropout_p_min)
        dropout_func_binary = dropout_func_binary_generator(dropout_func= dropout_func)

    constraint_func_generator = constraint_func_func_generator(state_action_uniques = state_action_uniques, amount_flex = amount_flex)
    less_constrained_constraints = {key: [] for key in more_constrained_constraints}
    for key, values in more_constrained_constraints.items():
        for value in values:
            less_constrained_constraints[value].append(key)

    def get_constraints(constraint_names):
        more_constrained_constraints_list = []
        less_constrained_constraints_list = []
        constraint_funcs = []
        for name in constraint_names:
            constraint_val = int(name[1])
            ctype = name[0]
            constraint_funcs.append(constraint_func_generator(constraint_val, ctype = ctype))
        
        for name in constraint_names:
            temp1 = []
            temp2 = []
            for item in more_constrained_constraints[name]:
                if item in constraint_names:
                    temp1.append(constraint_names.index(item))
            for item in less_constrained_constraints[name]:
                if item in constraint_names:
                    temp2.append(constraint_names.index(item))
            more_constrained_constraints_list.append(temp1)
            less_constrained_constraints_list.append(temp2)
        
        return constraint_funcs, more_constrained_constraints_list, less_constrained_constraints_list
        
    

    n_params = d + 1 + d2 + d3

    var_idxs = range(1, min(w+1, nA)+1)
    
    def sa_to_idx_var(s,a):
        return state_action_uniques[s][a]

    all_idxs = list(idx_state_action_dict.keys())

    def sa_to_idx_all(s,a):
        return state_action_idx_dict[s][a]

    var_action_idxs = []
    var_action_to_idx = {}
    i = 0
    for action in actions:
        var_action_to_idx[action] = {}
        for var in var_idxs:
            var_action_to_idx[action][var] = i
            var_action_idxs.append(i)
            i+=1
    
    def sa_to_idx_var_action(s,a):
        return var_action_to_idx[a][int(sa_to_idx_var(s,a))]

    def transition_func(x,a):
        return transitions[x][a]

    dropout_idx_funcs = (all_idxs, sa_to_idx_all, var_idxs, sa_to_idx_var, var_action_idxs, sa_to_idx_var_action)

    class Movielens_Env(object):
        def __init__(self):
            self.nA = nA
            self.nS = nS 
            self.transition_func = transition_func
            self.reward = reward
            self.reward_nonoise = reward_nonoise
            self.dropout_func_binary = dropout_func_binary
            self.reset_state = reset_state
            self.terminal_state = str([-2]*w)
            self.reset()

        def reset(self):
            self.state = self.reset_state
            return self.state

        def step(self, action):
            ns = self.transition_func(self.state, action)
            term = self.dropout_func_binary(self.state, action)
            r = self.reward(self.state, action)

            self.state = ns
            if term:
                ns = self.terminal_state
                self.reset()
            
            return r, ns, term, None

    movielens_env = Movielens_Env()


    return movielens_env, (nA, nS, states, actions, d, w, n_params, state_to_idx, reset_state, max_life, max_lav, var_x,transition_func, reward, reward_nonoise, dropout_func, dropout_func_binary, constraint_func_generator,reward_func_generator_constrained, dropout_idx_funcs, get_constraints)  


def constraint_func_func_generator(state_action_uniques, amount_flex):
    def constraint_func_generator(constraint, ctype = 'g'): #returns true if constraint is violated
        if ctype == 'g':
            def constraint_func(state, action):

                return state_action_uniques[state][action] < constraint - amount_flex[state]
        elif ctype == 'e':
            def constraint_func(state, action):
                return (state_action_uniques[state][action] < constraint - amount_flex[state]) or (state_action_uniques[state][action] > constraint)
        elif ctype == 'l':
            def constraint_func(state, action):
                return state_action_uniques[state][action] > constraint
        
        elif ctype == 'o':
            def constraint_func(state, action):
                return action== 4 or (state_action_uniques[state][action] < constraint - amount_flex[state]) or (state_action_uniques[state][action] > constraint)
        elif ctype == 't':
            def constraint_func(state, action):
                return action in [2, 4] or (state_action_uniques[state][action] < constraint - amount_flex[state]) or (state_action_uniques[state][action] > constraint)
        
        setattr(constraint_func, 'name', f'{ctype}{constraint}' )
        return constraint_func
    
    return constraint_func_generator


def dropout_func_func_generator0(state_action_uniques):
    def dropout_func_generator(dropout_p_var):
        def dropout_func(state, action):
            return dropout_p_var[int(state_action_uniques[state][action])]
        return dropout_func
    return dropout_func_generator

def dropout_func_func_generator1(state_action_uniques):
    def dropout_func_generator(dropout_p_var_act):
        def dropout_func(state, action):
            return dropout_p_var_act[action][int(state_action_uniques[state][action])]
        return dropout_func
    return dropout_func_generator

def dropout_func_func_generator2(state_action_uniques):
    def dropout_func_generator(dropout_p_min):
        def dropout_func(state, action):
            return dropout_p_min
        return dropout_func
    return dropout_func_generator

def dropout_func_binary_generator(dropout_func):
    def dropout_func_binary(state, action):
        dropout_p = dropout_func(state, action)
        return np.random.choice([False, True], p = [1- dropout_p, dropout_p])
    return dropout_func_binary

def state_unstr(str_state):
    #state = [int(s) for s in re.findall(r'\d+', str_state)] # little bit slower
    #state = [int(elt) for elt in str_state if elt.isdigit()] # fastest but works only if all elt are < 10
    state = list(map(int,re.findall(r'\d+', str_state)))
    return state

def default_x(state, action, d):
    state = state_unstr(state)
    recency = sum([1./(delta+1) for delta, elt in enumerate(state) if elt == action]) # compute decay
    x = np.array([recency**j for j in range(d+1)]).reshape(1, -1)
    return x
    
def var_x_generator(w, d2 = 0, d3 =0, genre_params_used = None, actions = None):
    if actions is not None:
        def var_x(state, action, d):
            state = state_unstr(state)
            variability = len(set(state + [action]))/float(w)
            n_state = sum(np.array(state) == action) + 1
            try:
                most_recent_idx = state.find(action)
            except:
                most_recent_idx = w
            x = np.array([1, 1/n_state, most_recent_idx/w, variability])
            x = x.reshape(1, -1)
            return x
        return var_x
    else:
        if d2 == 0 and d3 == 0:
            return default_x
        else:
            def var_x(state, action, d):
                state = state_unstr(state)
                recency = sum([1./(delta+1) for delta, elt in enumerate(state) if elt == action]) # compute decay
                variability = len(set(state + [action]))/float(w)
                recency2 = recency
                var_rec = variability*recency2
                x = np.array([recency**j for j in range(d+1)] + [variability**j for j in range(1, d2+1)] + [var_rec **j for j in range(1, d3+1)])
                x = x.reshape(1, -1)
                return x
            
            return var_x

def reward_func_generator(x_func, d, genre_params, noise_std = 0.1):
    def reward(state,a):
        X = x_func(state, a, d)
        n = 0
        if noise_std > 0:
            n = np.random.normal(0, noise_std)
            
        return genre_params[int(a)].dot(X.T)[0]+n
    return reward

def reward_func_generator_constrained_generator(x_func, d, genre_params):
    def reward_func_generator_constrained(constraint_func, violation_reward = 0, noise_std = 0.1):
        def reward(state,a):
            #if the constraint is violated
            if constraint_func(str(state), a):
                return violation_reward
            else:
                X = x_func(state, a, d)
                n = 0
                if noise_std > 0:
                    n = np.random.normal(0, noise_std)

                return genre_params[int(a)].dot(X.T)[0]+n
        return reward
    return reward_func_generator_constrained

def check_policy_generator(constraint_func, transition_func, constraint_percentage, t = 100):
    def check_policy(policy, s0):
        n_violations = 0
        s = s0
        for i in range(t):
            n_violations += constraint_func(s, policy[s])
            s = transition_func(s, policy[s])
        return n_violations/(t+3) > constraint_percentage, n_violations/(t+3)
    return check_policy

def no_constraint(state, a):
    return False