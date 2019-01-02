""" Hyperparameters for Box2d Point Mass task with PIGPS."""
from __future__ import division

import os.path
from datetime import datetime
import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.box2d.agent_box2d import AgentBox2D
from gps.agent.box2d.point_mass_world import PointMassWorld
from gps.algorithm.algorithm_pigps import AlgorithmPIGPS
from gps.algorithm.algorithm_pigps import AlgorithmMDGPS
from gps.algorithm.cost.cost_state_sparse import CostStateSparse
from gps.algorithm.cost.cost_action_sparse import CostActionSparse
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.policy_opt.policy_opt_tf import PolicyOptTf
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
#from gps.algorithm.policy_opt.policy_opt_caffe import PolicyOptCaffe
from gps.algorithm.traj_opt.traj_opt_pi2 import TrajOptPI2
from gps.algorithm.policy.policy_prior import PolicyPrior
from gps.algorithm.policy.lin_gauss_init import init_pd
from gps.proto.gps_pb2 import END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION
from gps.gui.config import generate_experiment_info
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy_opt.tf_model_example import tf_network

SENSOR_DIMS = {
    END_EFFECTOR_POINTS: 3,
    END_EFFECTOR_POINT_VELOCITIES: 3,
    ACTION: 2
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/box2d_pointmass_pigps_tf_sparse_example/'

common = {
    'experiment_name': 'box2d_pointmass_pigps_tf_sparse_example' + '_' + \
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
    'type': AgentBox2D,
    'target_state' : [5, 20, 0],
    "world" : PointMassWorld,
    'render' : False,
    'x0': [[0,5,0,0,0,0]],
    '''
    'x0': [np.array([0, 5, 0, 0, 0, 0]),
           np.array([0, 10, 0, 0, 0, 0]),
           np.array([10, 5, 0, 0, 0, 0]),
           np.array([10, 10, 0, 0, 0, 0]),
        ],
    '''
    'rk': 0,
    'dt': 0.05,
    'substeps': 1,
    'conditions': common['conditions'],
    'pos_body_idx': np.array([]),
    'pos_body_offset': np.array([]),
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
    'smooth_noise_var': 3.0,
}

algorithm = {
    'type': AlgorithmPIGPS,
    'conditions': common['conditions'],
    'policy_sample_mode': 'replace',
    'sample_on_policy': True,
}

algorithm['init_traj_distr'] = {
    'type': init_pd,
    'init_var': 1.0,
    'pos_gains': 0.0,
    'dQ': SENSOR_DIMS[ACTION],
    'dt': agent['dt'],
    'T': agent['T'],
}

action_cost = {
    'type': CostActionSparse,
    'wu': np.array([5e-5, 5e-5])
}

state_cost = {
    'type': CostStateSparse,
    'data_types' : {
        END_EFFECTOR_POINTS: {
            'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
            'target_state': agent["target_state"],
        },
    },
}

algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, state_cost],
    'weights': [1.0, 1.0],
}

algorithm['traj_opt'] = {
    'type': TrajOptPI2,
    'kl_threshold': 2.0,
    'covariance_damping': 2.0,
    'min_temperature': 0.001,
}

'''
algorithm['policy_opt'] = {
    'type': PolicyOptCaffe,
    'weights_file_prefix': EXP_DIR + 'policy',
    'taiterations': 10000,
    'network_arch_params': {
        'n_layers': 2,
        'dim_hidden': [20],
    },
}
'''

algorithm['policy_opt'] = {
    'type': PolicyOptTf,
    'network_params': {
        'obs_include': [END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'obs_vector_data': [END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES],
        'sensor_dims': SENSOR_DIMS,
    },
    'network_model': tf_network,
    'iterations': 1000,
    'weights_file_prefix': EXP_DIR + 'policy',
}


algorithm['policy_prior'] = {
    'type': PolicyPrior,
}

config = {
    'iterations': 20,
    'num_samples': 30,
    'common': common,
    'verbose_trials': 1,
    'verbose_policy_trials': 0,
    'agent': agent,
    'gui_on': False,
    'algorithm': algorithm,
    'random_seed': 0,
}

common['info'] = generate_experiment_info(config)
