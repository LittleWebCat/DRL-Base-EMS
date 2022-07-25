# Add this code to your Gym classic_control __init__ file which path is appear like \gym\envs\classic_control\__init__.py
# Just One code is allowed to added at the same time, can not add both  
# If you use discrete Env, please add
from gym.envs.classic_control.vehicle_discrete import VehicleEnv
# If you use continuous Env, please add
from gym.envs.classic_control.vehicle_continous import VehicleEnv
# 


# Add this code to  your gym Env __init__ file which path is appear like \gym\envs\__init__.py
register(
    id='Vehicle-v0',
    entry_point='gym.envs.classic_control:VehicleEnv',
    max_episode_steps=1370,
    reward_threshold=1370,
)
# 