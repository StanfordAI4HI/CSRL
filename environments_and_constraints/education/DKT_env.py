import json
import sys

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

sys.path.insert(0, 'education')
sys.path.insert(0, 'environments_and_constraints/education')

from DKT_model import get_custom_DKT_model


def convert_keys_to_int(d):
    new_dict = {}
    for k, v in d.items():
        try:
            new_key = int(k)
        except ValueError:
            new_key = k
        new_dict[new_key] = v
    return new_dict


def bits_list(n, val = None):
    bits = [int(x) for x in bin(n)[2:]]
    if val is None or len(bits) >= val:
        return bits
    else:
        return [0] * (val-len(bits)) + bits


    
class DKT_env(object):
    def __init__(self, student_model, best_model_weights, idx_to_problem, problem_to_idx, n_features, H = 100, ws = 10, 
                 thresh = 0.85):
        
        self.student_model = student_model
        self.ws = ws
        self.thresh = thresh

        self.student_model.load_weights(best_model_weights)
        self.idx_to_problem = idx_to_problem
        self.problem_to_idx = problem_to_idx
        self.H = 100
        self.n_actions = len(self.problem_to_idx)
        
        self.n_time_features = len(bits_list(self.H))
        self.n_state_features = self.n_actions + self.n_time_features
        self.n_features = n_features
        
        self.reset()
        
    
    def format_input(self, problem_seq, answer_seq):
        if len(problem_seq) == 0:
            return np.array([[[-1]*self.n_features]])
        x_seq = []
        l_seq = len(problem_seq)
        for problem, answer in zip(problem_seq, answer_seq):
            idx = problem *2 + answer
            x = np.zeros(self.n_features)
            x[idx] = 1
            x_seq.append(x)
        
        return tf.convert_to_tensor(np.array([x_seq]))
    
    def reset(self):
        self.problems = []
        self.answers = []
        self.h = 0
        self.problem_probs = [0.0 for i in range(self.n_actions)]
        self.mastered = {i: 0 for i in range(self.n_actions)}
        return np.array(self.problem_probs + bits_list(self.h, val = self.n_time_features))
    
    def step(self, problem_idx):
        problem = self.idx_to_problem[problem_idx]
        prob = 0
        num_attempts = 0
        while prob < self.thresh and self.h < self.H and num_attempts < self.ws:
            self.problems.append(problem)
            self.answers.append(1)
            prob = self.student_model.predict(self.format_input(self.problems, self.answers))[0][-1][problem]
            self.h += 1
            num_attempts += 1
            
        
        terminate = self.h == self.H
        
        reward = int((prob >= self.thresh) and (self.problem_probs[problem_idx] < self.thresh))
        self.problem_probs[problem_idx] = prob
        next_state = np.array(self.problem_probs + bits_list(self.h, val = self.n_time_features))
        
        return next_state, reward, terminate, None

def get_DKT_env(saved_data_folder = 'environments_and_constraints/education/saved_data', THRESH = 0.85, H = 100, ws = 10):
    load_model_weights = f"{saved_data_folder}/saved_model_weights/bestvalmodel"

    with open(f'{saved_data_folder}/idx_to_problem.json', 'r') as f:
        idx_to_problem = json.load(f)
    with open(f'{saved_data_folder}/problem_to_idx.json', 'r') as f:
        problem_to_idx = json.load(f)
    
    all_problems_file_name = f'{saved_data_folder}/all_problems.txt'
    all_problems = []

    with open(all_problems_file_name, 'r') as filehandle:
        for line in filehandle:
            problem = line[:-1] # remove linebreak which is the last character of the string
            all_problems.append(int(problem))

    n_problems = len(all_problems)
    n_features = 2*n_problems

    idx_to_problem = convert_keys_to_int(idx_to_problem)
    problem_to_idx = convert_keys_to_int(problem_to_idx)

    student_model = get_custom_DKT_model(saved_data_folder = saved_data_folder)

    return DKT_env(student_model, load_model_weights, idx_to_problem, problem_to_idx, n_features = n_features,
        H = H, ws = ws, thresh = THRESH)
    
