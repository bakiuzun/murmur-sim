from gymnasium.envs.registration import register 

register(id='uavenv',
         entry_point='envs.environement:UAVEnvironement',
         max_episode_steps=1500)


