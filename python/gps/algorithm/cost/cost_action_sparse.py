""" This file defines the torque (action) cost. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_ACTION
from gps.algorithm.cost.cost import Cost


class CostActionSparse(Cost):
    """ Computes torque penalties. """
    def __init__(self, hyperparams, base_dir):
        config = copy.deepcopy(COST_ACTION)
        config.update(hyperparams)
        Cost.__init__(self, config, base_dir)

    def eval(self, sample, iteration_num, sample_num):
        """
        Evaluate cost function and derivatives on a sample.
        Args:
            sample: A single sample
        """
        sample_u = sample.get_U()
        T = sample.T
        Du = sample.dU
        Dx = sample.dX
        l = 0.5 * np.sum(self._hyperparams['wu'] * (sample_u ** 2), axis=1)

        l[-1] = np.sum(l)
        l[:-1] = 0

        #l[:] = np.sum(l)
        #lu = self._hyperparams['wu'] * sample_u
        lu = np.zeros((T, Du))
        lx = np.zeros((T, Dx))
        luu = np.tile(np.diag(self._hyperparams['wu']), [T, 1, 1])
        lxx = np.zeros((T, Dx, Dx))
        lux = np.zeros((T, Du, Dx))
        return l, lx, lu, lxx, luu, lux
