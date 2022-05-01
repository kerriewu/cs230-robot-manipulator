from gym.envs.registration import register

register(
    id="gym_simulations/Discrete2DoF-v0",
    entry_point="gym_simulations.envs:Discrete2DoF",
    max_episode_steps=1000,
)
