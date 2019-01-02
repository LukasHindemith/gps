""" Hyperparameters for Baxter trajectory optimization experiment. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.baxter.agent_baxter import AgentBaxter
from gps.agent.baxter.ball_in_a_cup_sparse_world import BIACSparse_World
from gps.algorithm.algorithm_traj_opt_pi2 import AlgorithmTrajOptPI2
from gps.algorithm.cost.cost_state_user import CostStateUser
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.gui.config import generate_experiment_info

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, ACTION, SPARSE_COST

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    SPARSE_COST: 1,
    ACTION: 7,
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/baxter_ball_in_a_cup_PI2/'
#EXP_DIR = "/home/h1nd3mann/masterarbeit/results/closeCabinet/parameter_optimization/gps/"

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'target_filename': EXP_DIR + 'target.npz',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}


if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentBaxter,
    'world': BIACSparse_World,
    'target_state': np.array([0.14764565083397108, 0.36355344672884304, 1.3721458147635026, 1.4595827196729712, -0.39193209130472323, -0.826815644670238, 0.25080585881926515]),
    'rk': 0,
    'dt': 0.05,
    'conditions': common['conditions'],
    'x0': np.concatenate([[-0.4218447166684888, 0.04716990922747647, 0.055223308363874894, 1.193437052974852, -0.11006312153077843, -1.3357137710512241, 0.023009711818281205],np.zeros(8)]),
    'T': 30,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, SPARSE_COST],
    'obs_include': [],
    'limb': 'right',
    'rate': 10,
    'smooth_noise_var': 1.0,
}

algorithm = {
    'type': AlgorithmTrajOptPI2,
    'conditions': common['conditions'],
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 0.04,
    'pos_gains': 0.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
}




'''
action_cost = {
    'type': CostAction,
    'wu': np.ones(7),
}

state_cost = {
    'type': CostState,
    'data_types': {
        DOOR_ANGLE: {
            'wp': np.array([1.0]),
            'target_state': np.array([0.0]),
        },
        IR_RANGE: {
            'wp': np.array([1.0]),
            'target_state': np.array([0.0]),
        },

    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1e-5, 1.0],
}
'''

algorithm['cost'] = {
    'type': CostStateUser,
    'data_types': {
        SPARSE_COST: {
            'wp': np.array([1.0]),
            'target_state': np.array([5.0]),
        },
    }
}


algorithm['traj_opt'] = {
    'type': TrajOptPI2,
    'kl_threshold': 2.0,
    'covariance_damping': 1.5,
    'min_temperature': 0.001,
}

algorithm['policy_opt'] = {}

'''
algorithm['policy_prior'] = {
    'type': PolicyPriorGMM,
    'max_clusters': 20,
    'min_samples_per_cluster': 40,
    'max_samples': 40,
}
'''

config = {
    'iterations': 20,
    'num_samples': 20,
    'common': common,
    'verbose_trials': 20,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'dQ': algorithm['init_traj_distr']['dQ'],
    'random_seed': 2,
}

common['info'] = generate_experiment_info(config)
