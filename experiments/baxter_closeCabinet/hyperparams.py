""" Hyperparameters for Baxter trajectory optimization experiment. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.baxter.agent_baxter import AgentBaxter
from gps.agent.baxter.closeCabinet_world import CloseCabinetWorld
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.gui.config import generate_experiment_info

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, ACTION, DOOR_ANGLE, IR_RANGE

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    #END_EFFECTOR_POINTS: 3,
    DOOR_ANGLE: 1,
    IR_RANGE: 1,
    ACTION: 7,
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/baxter_closeCabinet/'

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
    'world': CloseCabinetWorld,
    'target_state': np.array([0.14764565083397108, 0.36355344672884304, 1.3721458147635026, 1.4595827196729712, -0.39193209130472323, -0.826815644670238, 0.25080585881926515]),
    'dt': 0.05,
    'conditions': common['conditions'],
    'x0': np.concatenate([[-0.5039126888203584, 0.35204859081970247, 0.9963205217315763, 1.1792477306869118, -3.049170311119231, -1.2517283229144975, 2.876597472482122],np.zeros(9)]),
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, DOOR_ANGLE, IR_RANGE],
    'obs_include': [],
    'limb': 'right',
    'openedAngle': -72.77,
    'closedAngle': -154.73,
    'rate': 10,
}


algorithm = {
    'type': AlgorithmTrajOpt,
    'conditions': common['conditions'],
}
algorithm['init_traj_distr'] = {
    'type': init_lqr,
    'init_gains':  np.zeros(SENSOR_DIMS[ACTION]),
    'init_acc': np.zeros(SENSOR_DIMS[ACTION]),
    'init_var': 0.003,
    'stiffness': 0.001,
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

"""
        JOINT_ANGLES: {
            'wp': np.ones(7),
            'target_state': agent["target_state"],
        },
"""


algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1.0, 2.0],
}
'''

algorithm['cost'] = {
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
    'l1': 1.0,
    'l2': 0.0001,
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
    'common': common,
    'verbose_trials': 0,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'num_samples': 5,
}

common['info'] = generate_experiment_info(config)
