""" This file defines an agent for the gym simulator."""

import numpy as np
from copy import deepcopy
from time import sleep

from gps.agent.agent import Agent
from gps.agent.agent_utils import generate_noise, setup
from gps.agent.config import AGENT_GYM
from gps.proto.gps_pb2 import ACTION
from gps.sample.sample import Sample
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS, REWARD, CARPOSITION, CARVELOCITY
import gym

class AgentGym(Agent):
  """
  All communication between the algorithm and Gym-Environments is done through this class.
  """

  def __init__(self, hyperparams):
    config = deepcopy(AGENT_GYM)
    config.update(hyperparams)
    Agent.__init__(self, config)

    self.x0 = self._hyperparams["x0"]
    self._env_name = self._hyperparams['env']
    self._setup_env(self._env_name)

  def _setup_conditions(self):
    """
    Helper method for setting some hyperparameters that may vary by
    condition.
    """
    conds = self._hyperparams['conditions']
    for field in ('x0', 'x0var', 'pos_body_idx', 'pos_body_offset',
                  'noisy_body_idx', 'noisy_body_var'):
      self._hyperparams[field] = setup(self._hyperparams[field], conds)

  def _setup_env(self, env):
    """
    Helper method for handling setup of the gym environment.
    """
    self._envs = []
    for i in range(self._hyperparams['conditions']):
      env = gym.make(env)
      env.reset()
      self._envs.append(env)

  def parse_result(self, result):
    tmp_state = result[0]
    if self._env_name == "BipedalWalker-v2":
      state = {JOINT_ANGLES: np.array([tmp_state[4],
                                        tmp_state[6],
                                        tmp_state[9],
                                        tmp_state[11]]),
               JOINT_VELOCITIES: np.array([tmp_state[5],
                                            tmp_state[7],
                                            tmp_state[10],
                                            tmp_state[12]]),
               REWARD: np.array(result[1]),
               END_EFFECTOR_POINTS: np.zeros(3),
               }

    if self._env_name == "MountainCarStatic-v0":
      state = {CARPOSITION: np.array([tmp_state[0]]),
               CARVELOCITY: np.array([tmp_state[1]]),
               REWARD: np.array(result[1]),
               END_EFFECTOR_POINTS: np.zeros(3),
      }

    done = result[2]

    return state, done

  def sample(self, policy, condition, verbose=False, save=True, noisy=True):
    """
    Runs a trial and constructs a new sample containing information
    about the trial.

    Args:
        policy: Policy to to used in the trial.
        condition (int): Which condition setup to run.
        verbose (boolean): Whether or not to plot the trial (not used here).
        save (boolean): Whether or not to store the trial into the samples.
        noisy (boolean): Whether or not to use noise during sampling.
    """
    if self._hyperparams['render']:
      self._envs[condition].render()
    state = self._envs[condition].reset()
    b2d_X = {CARPOSITION: np.array([state[0]]),
             CARVELOCITY: np.array([state[1]]),
             REWARD: np.array([0]),
             END_EFFECTOR_POINTS: np.zeros(3)}

    new_sample = self._init_sample(b2d_X)
    U = np.zeros([self.T, self.dU])
    if noisy:
      noise = generate_noise(self.T, self.dU, self._hyperparams)
    else:
      noise = np.zeros((self.T, self.dU))
    for t in range(self.T):
      X_t = new_sample.get_X(t=t)
      obs_t = new_sample.get_obs(t=t)
      U[t, :] = policy.act(X_t, obs_t, t, noise[t, :])
      if (t + 1) < self.T:
        if self._hyperparams['render']:
          self._envs[condition].render()
          sleep(1.0/self._hyperparams['rate'])
        try:
          result = self._envs[condition].step(U[t, :])
          b2d_X, done = self.parse_result(result)
          self._set_sample(new_sample, b2d_X, t)
          #print(b2d_X[CARPOSITION])
          if done:
            for t_left in range(self.T-t-1):
              self._set_sample(new_sample, b2d_X, t+t_left)
            break
        except:
          print(U[t,:])

    #print(U[-1,:], noise[-1,:])
    new_sample.set(ACTION, U)
    if save:
      self._samples[condition].append(new_sample)
    return new_sample

  def _init_sample(self, b2d_X):
    """
    Construct a new sample and fill in the first time step.
    """
    sample = Sample(self)
    self._set_sample(sample, b2d_X, -1)
    return sample

  def _set_sample(self, sample, b2d_X, t):
    for sensor in b2d_X.keys():
      sample.set(sensor, np.array(b2d_X[sensor]), t=t + 1)