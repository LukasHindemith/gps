""" This file defines the PI2-based trajectory optimization method. """
import copy
import numpy as np

from gps.algorithm.algorithm import Algorithm
from gps.algorithm.config import ALG_PI2


class AlgorithmTrajOptPI2(Algorithm):
    """ Sample-based trajectory optimization with PI2. """
    def __init__(self, hyperparams, base_dir):
        config = copy.deepcopy(ALG_PI2)
        config.update(hyperparams)
        Algorithm.__init__(self, config, base_dir)


    def iteration(self, sample_lists, iteration_num):
        """
        Run iteration of PI2.
        Args:
            sample_lists: List of SampleList objects for each condition.
        """
        # Copy samples for all conditions.
        for m in range(self.M):
            self.cur[m].sample_list = sample_lists[m]
            for i, sample in enumerate(self.cur[m].sample_list):
                np.savetxt("{}/update{}_sample{}_trajectory.txt".format(self._base_dir, iteration_num, i), sample.get_U())

        # Evaluate cost function for all conditions and samples.
        for m in range(self.M):
            self._eval_cost(m, iteration_num)

        # Run inner loop to compute new policies.
        for _ in range(self._hyperparams['inner_iterations']):
            self._update_trajectories()

        self._advance_iteration_variables()
