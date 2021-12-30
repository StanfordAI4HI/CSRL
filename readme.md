# CSRL

Implementation of CSRL from the AAAI2022 paper: Constraint Sampling Reinforcement Learning: Incorporating Expertise For
Faster Learning

Python: 3.6.13\
Includes implementations in both tensorflow and pytorch.\
See requirements.txt for required packages.

Description:\
See run_experiment.ipynb to run the experiments in each domain.\
The environemnts used as well as a notebook detailing how to use them are in the "environments_and_constraints" directory.\
The agent implementations used are in the "agents" directory.\
The "fit_environments" directory contains code for fitting the parameters of the various environments.

Todos:
- Add UCRL agent and Movielens experiment code

Credits:\
The rainbow implementation is based off of the rainbow tutorial: https://github.com/Curt-Park/rainbow-is-all-you-need and the segment tree class (segment_tree.py) is also their implementation.
The HIV environment is from https://bitbucket.org/rlpy/rlpy/src/master/rlpy/Domains/HIVTreatment.py
