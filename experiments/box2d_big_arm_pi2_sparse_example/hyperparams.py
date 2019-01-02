""" Hyperparameters for Box2d Point Mass."""
from __future__ import division

import os.path
from datetime import datetime
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.box2d.agent_box2d import AgentBox2D
from gps.agent.box2d.big_arm_world import BigArmWorld
from gps.algorithm.algorithm_pigps import AlgorithmPIGPS
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.algorithm_traj_opt_pi2 import AlgorithmTrajOptPI2
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.cost.cost_state_sparse import CostStateSparse
from gps.algorithm.cost.cost_action_sparse import CostActionSparse
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.gui.config import generate_experiment_info
from gps.algorithm.policy_opt.tf_model_example import tf_network

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, ACTION

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 3,
    ACTION: 7
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/box2d_big_arm_pi2_sparse_example/'
#EXP_DIR = "/home/h1nd3mann/masterarbeit/results/big_arm_example/parameter_optimization/gps_pi2/"

common = {
    'experiment_name': 'box2d_big_arm_pi2_sparse_example' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}

if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentBox2D,
    'target_state': np.zeros(7),
    'world': BigArmWorld,
    'render': False,
    'x0': np.concatenate([[0.75*np.pi, -0.1*np.pi, -0.1*np.pi, -0.1*np.pi, -0.1*np.pi, -0.1*np.pi, -0.1*np.pi],np.zeros(10)]),
    'rk': 0,
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS],
    'obs_include': [],
    'smooth_noise_var': 3.0,
}

algorithm = {
    'type': AlgorithmTrajOptPI2,
    'conditions': common['conditions'],
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 0.1,
    'pos_gains': 0.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
}


action_cost = {
    'type': CostActionSparse,
    'wu': np.ones(7),
}

state_cost = {
    'type': CostStateSparse,
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.ones(7),
            'target_state': agent["target_state"],
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
    'data_types' : {
        JOINT_ANGLES: {
            'wp': np.array([1, 1]),
            'target_state': agent["target_state"],
        },
    },
    'wp_final_multiplier': 10.0,
}
'''

algorithm['traj_opt'] = {
    'type': TrajOptPI2,
    'kl_threshold': 2.0,
    'covariance_damping': 2.0,
    'min_temperature': 0.001,
}

'''
algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES,END_EFFECTOR_POINTS],
        'obs_vector_data': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS],
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': tf_network,
    'iterations': 1000,
    'weights_file_prefix': EXP_DIR + 'policy',
}


algorithm['policy_prior'] = {
    'type': PolicyPrior,
}
'''

config = {
    'iterations': 40,
    'num_samples': 30,
    'verbose_trials': 0,
    'common': common,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    'dQ': algorithm['init_traj_distr']['dQ'],
    'random_seed': 9,
}

common['info'] = generate_experiment_info(config)
