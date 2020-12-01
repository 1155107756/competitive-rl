import os

import gym
import matplotlib.pyplot as plt
from stable_baselines.common.cmd_util import make_atari_env

import competitive_rl

competitive_rl.register_competitive_envs()

pong_single_env = gym.make("cPong-v0")

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2, results_plotter

env = DummyVecEnv([lambda: gym.make("cPong-v0")])
#env = make_atari_env('cPong-v0', num_env=4, seed=0)
# Automatically normalize the input features and reward
'''env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)'''

model = PPO2('CnnPolicy', env, n_steps=128,noptepochs=4,nminibatches=4,learning_rate=2.5e-4, cliprange=0.1,vf_coef=0.5,
             ent_coef=0.01, cliprange_vf=-1,verbose=1,tensorboard_log="./a2c_cartpole_tensorboard/")
model.learn(total_timesteps=1000000)

# Don't forget to save the VecNormalize statistics when saving the agent
log_dir = "./tmp/"
model.save(log_dir + "ppo_pong")
# stats_path = os.path.join(log_dir, "vec_normalize.pkl")
#env.save(stats_path)

# results_plotter.plot_results([log_dir], 1e7, results_plotter.X_TIMESTEPS, "ppo_pong")
# plt.show()

# To demonstrate loading
del model, env

