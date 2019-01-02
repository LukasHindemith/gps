from gym.envs.registration import register

register(
  id='MountainCarStatic-v0',
  entry_point='gps.agent.gym.mountaincar_static:MountainCarStatic',
)