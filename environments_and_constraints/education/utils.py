import json
import sys
sys.path.insert(0, 'agents')

import numpy as np

from cDDQN_tf import DQNLoadedPolicy

def convert_keys_to_int(d):
    new_dict = {}
    for k, v in d.items():
        try:
            new_key = int(k)
        except ValueError:
            new_key = k
        new_dict[new_key] = v
    return new_dict

def constraint_func_generator(prereq_graph, THRESH, n_actions, problem_to_idx):
    def constraint_func(state):
        if prereq_graph == None:
            return range(n_actions)
        else:
            mastery = state[0:n_actions] > THRESH
            available_problems_idxs = []
            for problem, prereqs in prereq_graph.items():
                problem_idx = problem_to_idx[problem]
                if not mastery[problem_idx]:
                    prereqs_mastered = True
                    for prereq in prereqs:
                        if not mastery[problem_to_idx[prereq]]:
                            prereqs_mastered = False
                    if prereqs_mastered:
                        available_problems_idxs.append(problem_idx)
                        
            return available_problems_idxs
    return constraint_func

def get_info(constraints_names, problem_to_idx, THRESH = 0.85, n_actions = 51):
    #Constraints Info:

    epsilon_decays = {
        'none': 0.9997,
        'C8': 0.999,
        'C30':0.999,
        'C50':0.999,
        'C55':0.999,
        'C60':0.999,
        'C65':0.999,
        'C70':0.999,
        'C75':0.999,
        'C80':0.999,
        'C85':0.999,
        'C90':0.999,
        'C95':0.999,
        'C100':0.999,
        'C133': 0.999
    }

    more_constrained_constraints = {
        'none':['C8', 'C30', 'C50', 'C55','C60', 'C65','C70', 'C75','C80', 'C85','C90', 'C95', 'C100', 'C133'],
        'C8':  ['C30', 'C50', 'C55','C60', 'C65','C70', 'C75','C80', 'C85','C90', 'C95', 'C100', 'C133'],
        'C30': ['C50', 'C55','C60', 'C65','C70', 'C75','C80', 'C85','C90', 'C95', 'C100', 'C133'],
        'C50': ['C55','C60', 'C65','C70', 'C75','C80', 'C85','C90', 'C95', 'C100', 'C133'],
        'C55': ['C60','C65','C70', 'C75','C80', 'C85','C90', 'C95', 'C100', 'C133'],
        'C60': ['C65','C70', 'C75','C80', 'C85','C90', 'C95', 'C100', 'C133'],
        'C65': ['C70','C75','C80', 'C85','C90', 'C95', 'C100', 'C133'],
        'C70': ['C75','C80', 'C85','C90', 'C95', 'C100', 'C133'],
        'C75': ['C80','C85','C90', 'C95', 'C100', 'C133'],
        'C80': ['C85','C90', 'C95', 'C100', 'C133'],
        'C85': ['C90','C95', 'C100', 'C133'],
        'C90': ['C95','C100', 'C133'],
        'C95': ['C100', 'C133'],
        'C100':['C133'],
        'C133':[],
    }

    less_constrained_constraints = {key: [] for key in more_constrained_constraints}
    for key, values in more_constrained_constraints.items():
        for value in values:
            less_constrained_constraints[value].append(key)

    
    constraint_func_list = []
    for name in constraints_names:
        if name != 'none':
            load_file = f'environments_and_constraints/education/graph_structures/prereq_graph_{name}.json'
            with open(load_file, 'r') as f:
                prereq_graph = json.load(f)
                    
            prereq_graph = convert_keys_to_int(prereq_graph)
            constraint_func_list.append(constraint_func_generator(prereq_graph, THRESH, n_actions, problem_to_idx))

        else:
            constraint_func_list.append(constraint_func_generator(None, THRESH, n_actions, problem_to_idx))
    
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