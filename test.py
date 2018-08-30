import gym
import random
import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv



char_list = list('SFFFFFFFFFFFFFFG')
for i in range(2):
    char_list[random.randint(1, 14)] = 'H'
my_map = [''.join(char_list[i:i+4]) for i in [0, 4, 8, 12]]
env = FrozenLakeEnv(desc = np.asarray(my_map, dtype='c'), is_slippery= False)
env = env.unwrapped



for i in range(10):
    b = env.render()
    a = env.step(1)
    print(a)

