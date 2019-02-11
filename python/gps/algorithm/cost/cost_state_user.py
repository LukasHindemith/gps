""" This file defines the human feedback cost. """
import copy

import numpy as np

from gps.algorithm.cost.config import COST_STATE
from gps.algorithm.cost.cost import Cost
from gps.algorithm.cost.cost_utils import get_ramp_multiplier


class CostStateUser(Cost):
    """ Computes the human feedback 5 star cost. """
    def __init__(self, hyperparams, base_dir):
        config = copy.deepcopy(COST_STATE)
        config.update(hyperparams)
        Cost.__init__(self, config, base_dir)

    def eval(self, sample, iteration_num, sample_num):
        """
        Evaluates the human feedback and calculates sparse cost.
        Args:
            sample:  A single sample
        """
        T = sample.T
        Du = sample.dU
        Dx = sample.dX

        final_l = np.zeros(T)
        final_lu = np.zeros((T, Du))
        final_lx = np.zeros((T, Dx))
        final_luu = np.zeros((T, Du, Du))
        final_lxx = np.zeros((T, Dx, Dx))
        final_lux = np.zeros((T, Du, Dx))

        for data_type in self._hyperparams['data_types']:
            config = self._hyperparams['data_types'][data_type]
            wp = config['wp']
            tgt = config['target_state']
            x = sample.get(data_type)
            _, dim_sensor = x.shape

            wpm = get_ramp_multiplier(
                self._hyperparams['ramp_option'], T,
                wp_final_multiplier=self._hyperparams['wp_final_multiplier']
            )
            wp = wp * np.expand_dims(wpm, axis=-1)
            # Compute 5-star system cost value and calculate sparse cost.
            dist = tgt - x
            dist[:-1] = 0.0
            np.savetxt("{}/update{}_sample{}_{}_dists.txt".format(self._base_dir, iteration_num, sample_num, data_type), np.array(dist))

            l = np.concatenate(dist)
            ls = np.zeros((T, 1))
            lss = np.zeros((T, 1, 1))

            final_l += l

            sample.agent.pack_data_x(final_lx, ls, data_types=[data_type])
            sample.agent.pack_data_x(final_lxx, lss,
                                     data_types=[data_type, data_type])
        return final_l, final_lx, final_lu, final_lxx, final_luu, final_lux
