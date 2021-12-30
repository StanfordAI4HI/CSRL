import pickle
import os
from time import time
import json
from tqdm.notebook import tqdm
import random
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../agents')

import torch
import numpy as np
import gym

from agents.cDDQN_pytorch import *


save_model_paths = {
    0: 'p0',
    1: 'p1',
    2: 'p2',
    3: 'p3',
    4: 'p4',
    5: 'p5',
    6: 'p6',
    7: 'p7',
}

policies_needed = [0, 1, 2, 3, 4, 5, 6, 7]

# Constraints Info:
constraint_name_to_lists = {
    'none': (None, None),
    'p0_1': ([0], [1]),
    'p1_1': ([1], [1]),
    'p2_1': ([2], [1]),
    'p3_1': ([3], [1]),
    'p4_1': ([4], [1]),
    'p5_1': ([5], [1]),
    'p6_1': ([6], [1]),
    'p7_1': ([7], [1]),
    'p01_1': ([0, 1], [1]),
    'p23_1': ([2, 3], [1]),
    'p45_1': ([4, 5], [1]),
    'p67_1': ([6, 7], [1]),
    'p012_1': ([0, 1, 2], [1]),
    'p01234567_1': ([0, 1, 2, 3, 4, 5, 6, 7], [1]),
}

epsilon_decays = {
    'none': 0.999,
    'p0_1': 0.99,
    'p1_1': 0.99,
    'p2_1': 0.99,
    'p3_1': 0.99,
    'p4_1': 0.99,
    'p5_1': 0.99,
    'p6_1': 0.99,
    'p7_1': 0.99,
    'p01_1': 0.99,
    'p23_1': 0.99,
    'p45_1': 0.99,
    'p67_1': 0.99,
    'p012_1': 0.995,
    'p01234567_1': 0.995,
}

more_constrained_constraints = {
    'none': ['p0_1', 'p1_1', 'p2_1', 'p3_1', 'p4_1', 'p5_1', 'p6_1', 'p7_1', 'p01_1', 'p23_1', 'p45_1', 'p67_1',
             'p012_1', 'p01234567_1'],
    'p0_1': [],
    'p1_1': [],
    'p2_1': [],
    'p3_1': [],
    'p4_1': [],
    'p5_1': [],
    'p6_1': [],
    'p7_1': [],
    'p01_1': ['p0_1', 'p1_1'],
    'p23_1': ['p2_1', 'p3_1'],
    'p45_1': ['p4_1', 'p5_1'],
    'p67_1': ['p6_1', 'p7_1'],
    'p012_1': ['p0_1', 'p1_1', 'p2_1', 'p01_1'],
    'p01234567_1': ['p0_1', 'p1_1', 'p2_1', 'p3_1', 'p4_1', 'p5_1', 'p6_1', 'p7_1', 'p01_1', 'p012_1', 'p45_1', 'p67_1',
                    'p23_1'],
}



less_constrained_constraints = {key: [] for key in more_constrained_constraints}
for key, values in more_constrained_constraints.items():
    for value in values:
        less_constrained_constraints[value].append(key)


def loaded_policy_constraint_func_generator(loaded_policies_idx, n_allowed, policies_dict):
    def constraint_func(state):
        if loaded_policies_idx is not None:
            allowed_actions = set()
            for pidx in loaded_policies_idx:
                allowed_actions.update(policies_dict[pidx].get_actions(state, n_a=n_allowed[0]))
            return list(allowed_actions)
        else:
            return [0, 1, 2, 3]

    return constraint_func


def get_info(constraints_names, constraint_name_to_lists, epsilon_decays, more_constrained_constraints,
             less_constrained_constraints, policies_dict):
    constraint_func_list = []
    for name in constraints_names:
        pidx, pnuma = constraint_name_to_lists[name]
        constraint_func_list.append(loaded_policy_constraint_func_generator(pidx, pnuma, policies_dict))

    epsilon_decay_list = [epsilon_decays[name] for name in constraints_names]
    more_constrained_constraints_list = []
    less_constrained_constraints_list = []
    for name in constraints_names:
        temp1 = []
        temp2 = []
        for item in more_constrained_constraints[name]:
            if item in constraints_names:
                temp1.append(constraints_names.index(item))
        for item in less_constrained_constraints[name]:
            if item in constraints_names:
                temp2.append(constraints_names.index(item))
        more_constrained_constraints_list.append(temp1)
        less_constrained_constraints_list.append(temp2)
    return constraint_func_list, epsilon_decay_list, more_constrained_constraints_list, less_constrained_constraints_list