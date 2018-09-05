"""This file definces an agent for the Baxter Robot."""

import copy
import time
import numpy as np

import rospy

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_BAXTER
from gps.sample.sample_list import SampleList
from gps.sample.sample import Sample
from gps.proto.gps_pb2 import ACTION
from gps.agent.baxter.baxter_arm import BaxterArm

try:
    from gps.algorithm.policy.tf_policy import TfPolicy
except ImportError:  # user does not have tf installed.
    print("user doesn't have tf installed.")
    TfPolicy = None

class AgentBaxter(Agent):
    """
    All communication between the algorithms and Baxter is done through
    this class.
    """
    def __init__(self, hyperparams, init_node=True):
        """
        Initialize the agent.
        Args:
            hyperparams: Dictionary of hyperparameters.
            init_node: Whether or not to initialize a new ROS node.
        """

        config = copy.deepcopy(AGENT_BAXTER)
        config.update(hyperparams)
        Agent.__init__(self, config)
        if init_node:
            rospy.init_node('gps_agent_baxter_node')

        conditions = self._hyperparams['conditions']

        '''
        for field in ('x0'):
            self._hyperparams[field] = setup(self._hyperparams[field],
                                             conditions)
        '''
        self.x0 = self._hyperparams['x0']

        self.limb = self._hyperparams['limb']

        self.arm = BaxterArm(self.limb, self.x0)
        rospy.on_shutdown(self.arm.clean_shutdown)

    def sample(self, policy, condition, verbose=True, save=True, noisy=True):
        print("sample function:")

        self.arm.reset_arm()
        initial_state = self.arm.get_state()
        new_sample = self._init_sample(initial_state)
        U = np.zeros([self.T, self.dU])

        # Generate noise.
        if noisy:
            noise = generate_noise(self.T, self.dU, self._hyperparams)
        else:
            noise = np.zeros((self.T, self.dU))

        for t in range(self.T):
            X_t = new_sample.get_X(t=t)
            obs_t = new_sample.get_obs(t=t)
            U[t,:] = policy.act(X_t, obs_t, t, noise[t,:])
            if (t+1) < self.T:
                self.arm.run_next(U[t, :])
                current_state = self.arm.get_state()
                self._set_sample(new_sample, current_state, t)
        new_sample.set(ACTION, U)
        if save:
            self._samples[condition].append(new_sample)
        return new_sample

    def _init_sample(self, initial_state):
        """
        Construct a new sample and fill in the first time step
        :param initial_state:
        :return:
        """
        sample = Sample(self)
        self._set_sample(sample, initial_state, -1)
        return sample

    def _set_sample(self, sample, state, t):
        for sensor in state.keys():
            sample.set(sensor, np.array(state[sensor]), t=t+1)
