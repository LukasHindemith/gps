""" Hyperparameters for Baxter trajectory optimization experiment. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.baxter.agent_baxter import AgentBaxter
from gps.agent.baxter.closeCabinet_world import CloseCabinetWorld
from gps.algorithm.algorithm_traj_opt_pi2 import AlgorithmTrajOptPI2
from gps.algorithm.cost.cost_state import CostState
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
EXP_DIR = BASE_DIR + '/../experiments/baxter_closeCabinet_PI2/'
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
    'world': CloseCabinetWorld,
    'target_state': np.array([0.14764565083397108, 0.36355344672884304, 1.3721458147635026, 1.4595827196729712, -0.39193209130472323, -0.826815644670238, 0.25080585881926515]),
    'rk': 0,
    'dt': 0.05,
    'conditions': common['conditions'],
    'x0': np.concatenate([[-0.381961216183468, 0.3336408213650775, 1.270519587566094, 0.9955535313376335, -0.23239808936464018, 1.4289031039152629, -0.15224759319762732],np.zeros(9)]),
    'T': 50,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, DOOR_ANGLE, IR_RANGE],
    'obs_include': [],
    'limb': 'right',
    'openedAngle': -139.90,
    'closedAngle': 136.04,
    'rate': 10,
    'smooth_noise_var': 3.0,
}

algorithm = {
    'type': AlgorithmTrajOptPI2,
    'conditions': common['conditions'],
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 0.002,
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
    }
}


algorithm['traj_opt'] = {
    'type': TrajOptPI2,
    'kl_threshold': 2.0,
    'covariance_damping': 1.0,
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
    'iterations': 10,
    'num_samples': 30,
    'common': common,
    'verbose_trials': 10,
    'agent': agent,
    'gui_on': True,
    'algorithm': algorithm,
    'dQ': algorithm['init_traj_distr']['dQ'],
    'random_seed': 0,
}

common['info'] = generate_experiment_info(config)
