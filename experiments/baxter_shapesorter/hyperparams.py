""" Hyperparameters for Baxter trajectory optimization experiment. """
from __future__ import division

from datetime import datetime
import os.path

import numpy as np

from gps import __file__ as gps_filepath
from gps.agent.baxter.agent_baxter import AgentBaxter
from gps.agent.baxter.shapesorter_world import ShapesorterWorld
from gps.algorithm.cost.cost_state import CostState
from gps.algorithm.cost.cost_action import CostAction
from gps.algorithm.cost.cost_fk import CostFK
from gps.algorithm.cost.cost_sum import CostSum
from gps.algorithm.cost.cost_utils import RAMP_LINEAR, RAMP_FINAL_ONLY
from gps.algorithm.algorithm_traj_opt import AlgorithmTrajOpt
from gps.algorithm.dynamics.dynamics_lr_prior import DynamicsLRPrior
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM
from gps.algorithm.traj_opt.traj_opt_lqr_python import TrajOptLQRPython
from gps.algorithm.policy.lin_gauss_init import init_lqr
from gps.algorithm.policy.policy_prior_gmm import PolicyPriorGMM
from gps.gui.config import generate_experiment_info

from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, ACTION

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 7,
    ACTION: 7,
}

BASE_DIR = '/'.join(str.split(gps_filepath, '/')[:-2])
EXP_DIR = BASE_DIR + '/../experiments/baxter_shapesorter/'

common = {
    'experiment_name': 'my_experiment' + '_' + \
            datetime.strftime(datetime.now(), '%m-%d-%y_%H-%M'),
    'experiment_dir': EXP_DIR,
    'data_files_dir': EXP_DIR + 'data_files/',
    'log_filename': EXP_DIR + 'log.txt',
    'conditions': 1,
}


if not os.path.exists(common['data_files_dir']):
    os.makedirs(common['data_files_dir'])

agent = {
    'type': AgentBaxter,
    'world': ShapesorterWorld,
    'ee_tgt': np.array([0.7393028527033758, -0.14282140960308415, -0.26466171209831924, 0.7543105321666994, 0.6561052355532673, 0.014279957411678743, -0.018374541036531588]),
    'dt': 0.05,
    'conditions': common['conditions'],
    'x0': np.concatenate([[0.3643204371227858, -1.2394564766114142, 0.2676796474860047, 1.6923643042345826, -0.05483981316690354, 1.1075341288532687, 1.8852623883111734],np.zeros(14)]),
    'T': 50,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS],
    'obs_include': [],
    'limb': 'right',
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
    'init_var': 0.005,
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

'''
algorithm['cost'] = {
    'type': CostState,
    'data_types': {
        JOINT_ANGLES: {
            'wp': np.ones(7),
            'target_state': agent['target_state'],
        },
    },
}
'''

algorithm['cost'] = {
    'type': CostState,
    'data_types': {
        END_EFFECTOR_POINTS: {
            'wp': np.ones(7),
            'target_state': agent['ee_tgt'],
        },
    },
    'l1': 1.0,
    'l2': 0.0001,
    'ramp_option': RAMP_LINEAR,
    'wp_final_multiplier': 10.0,
}



action_cost = {
    'type': CostAction,
    'wu': np.array([0.001,0.001,0.001,0.001,0.001,0.001,0.001]),
}
'''
algorithm['cost'] = {
    'type': CostFK,
    'target_end_effector': agent['ee_tgt'],
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 1.0,
    'l2': 0.0001,
    'alpha': 1.0,
    'ramp_option': RAMP_LINEAR,
}
'''

fk_cost2 = {
    'type': CostFK,
    'target_end_effector': agent['ee_tgt'],
    'wp': np.ones(SENSOR_DIMS[END_EFFECTOR_POINTS]),
    'l1': 1.0,
    'l2': 0.0,
    'alpha': 1.0,
    'wp_final_multiplier': 10.0,  # Weight multiplier on final timestep.
    'ramp_option': RAMP_FINAL_ONLY,
}

'''
algorithm['cost'] = {
    'type': CostSum,
    'costs': [action_cost, fk_cost1],
    'weights': [1.0, 1.0],
}
'''

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
