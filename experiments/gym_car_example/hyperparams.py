""" Hyperparameters for Box2d Point Mass."""
from __future__ import division

import os.path
from datetime import datetime
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.gym.agent_gym import AgentGym
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.gui.config import generate_experiment_info
from gps.proto.gps_pb2 import CARPOSITION, CARVELOCITY, REWARD, END_EFFECTOR_POINTS, ACTION

SENSOR_DIMS = {
    CARPOSITION: 1,
    CARVELOCITY: 1,
    REWARD: 1,
    END_EFFECTOR_POINTS: 3,
    ACTION: 1
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
#EXP_DIR = BASE_DIR + '/../experiments/gym_car_example/'
EXP_DIR = '/home/h1nd3mann/masterarbeit/results/mountaincar/parameter_optimization/gps/'


common = {
    'experiment_name': 'gym_car_example' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentGym,
    'target_state': np.array([0.45]),
    'env': 'MountainCarStatic-v0',
    'render': False,
    'rate': 1000,
    'x0': np.zeros(6),
    'rk': 0,
    'dt': 0.05,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'T': 500,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [CARPOSITION, CARVELOCITY, REWARD, END_EFFECTOR_POINTS],
    'obs_include': [],
}

algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
}

algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains': np.zeros(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 1.0,
    'stiffness': 0.01,
    'dt': agent['dt'],
    'T': agent['T'],
}

state_cost= {
    'type': CostState,
    'data_types': {
        CARPOSITION: {
            'wp': np.array([1]),
            'target_state': agent["target_state"],
        },
    },
}

action_cost = {
    'type': CostAction,
    'wu': np.array([1])
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1e-5, 1.0],
}


algorithm['dynamics'] = {
    'type': DynamicsLRPrior,
    'regularization': 1e-6,
    'prior': {
        'type': DynamicsPriorGMM,
        'max_clusters': 20,
        'min_samples_per_cluster': 40,
        'max_samples': 20,
    },
}

algorithm['traj_opt'] = {
    'type': TrajOptLQRPython,
}

algorithm['policy_opt'] = {}

config = {
    'iterations': 20,
    'num_samples': 15,
    'verbose_trials': 15,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    'random_seed': 3,
}

common['info'] = generate_experiment_info(config)
