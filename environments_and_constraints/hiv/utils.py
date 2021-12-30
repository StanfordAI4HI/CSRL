import numpy as np
import sys

sys.path.insert(0, 'agents')
from cDDQN_tf import DQNLoadedPolicy


def get_info(constraints_names):
    save_model_paths = {
        0: 'hiv_saved_models_used/model0', 
        1: 'hiv_saved_models_used/model1',
        2: 'hiv_saved_models_used/model2', 
        3: 'hiv_saved_models_used/model3',
        4: 'hiv_saved_models_used/model4',
        5: 'hiv_saved_models_used/model5', 
        -1:'hiv_saved_models_used/modelNone'
    }
    policies_needed = [-1, 2, 4]
    policies = {key: DQNLoadedPolicy(4, 'environments_and_constraints/hiv/'+save_model_paths[key]) for key in policies_needed}

    def constraint_func_generator(policies_idx_list = [], optimal_action_number_list = []):
        def constraint_func(state):
            if policies_idx_list is not None:
                allowed_actions = set()
                for idx, policy_idx in enumerate(policies_idx_list):
                    _, actions_list = policies[policy_idx].get_action_list(state)
                    allowed_actions.update(actions_list[0:optimal_action_number_list[idx]])
                return list(allowed_actions)
            else:
                return [0, 1, 2, 3]
        return constraint_func

    constraint_name_to_lists = {
        'none': (None, None),
        'pNone_2':([-1], [2]),
        'p0_2':([0], [2]),
        'p1_2':([1], [2]),
        'p2_2':([2], [2]),
        'p3_2':([3], [2]),
        'p4_2':([4], [2]),
        'p5_2':([5], [2]),
        'pNone_1':([-1], [1]),
        'p0_1':([0], [1]),
        'p1_1':([1], [1]),
        'p2_1':([2], [1]),
        'p3_1':([3], [1]),
        'p4_1':([4], [1]),
        'p5_1':([5], [1]) 
    }

    epsilon_decays = {
        'none': 0.9999,
        'pNone_2':0.99,
        'p0_2':0.99,
        'p1_2':0.99,
        'p2_2':0.99,
        'p3_2':0.99,
        'p4_2':0.99,
        'p5_2':0.99,
        'pNone_1':0,
        'p0_1':0,
        'p1_1':0,
        'p2_1':0,
        'p3_1':0,
        'p4_1':0,
        'p5_1':0
    }

    more_constrained_constraints = {
        'none': ['pNone_2', 'p0_2', 'p1_2', 'p2_2', 'p3_2', 'p4_2', 'p5_2', 'pNone_1', 'p0_1', 'p1_1', 'p2_1', 'p3_1', 'p4_1', 'p5_1'],
        'pNone_2':['pNone_1'],
        'p0_2':['p0_1'],
        'p1_2':['p1_1'],
        'p2_2':['p2_1'],
        'p3_2':['p3_1'],
        'p4_2':['p4_1'],
        'p5_2':['p5_1'],
        'pNone_1':[],
        'p0_1':[],
        'p1_1':[],
        'p2_1':[],
        'p3_1':[],
        'p4_1':[],
        'p5_1':[]
    }

    less_constrained_constraints = {key: [] for key in more_constrained_constraints}
    for key, values in more_constrained_constraints.items():
        for value in values:
            less_constrained_constraints[value].append(key)

    constraint_func_list = []
    for name in constraints_names:
        pidx, pnuma = constraint_name_to_lists[name]
        constraint_func_list.append(constraint_func_generator(pidx, pnuma))
    
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