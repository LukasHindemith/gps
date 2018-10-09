""" This file defines the base cost class. """
import abc


class Cost(object):
    """ Cost superclass. """
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyperparams, base_dir):
        self._hyperparams = hyperparams
        self._base_dir = base_dir

    @abc.abstractmethod
    def eval(self, sample, iteration_num, sample_num):
        """
        Evaluate cost function and derivatives.
        Args:
            sample:  A single sample.
        """
        raise NotImplementedError("Must be implemented in subclass.")
