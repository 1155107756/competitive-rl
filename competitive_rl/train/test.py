# Load the agent
import gym
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

import competitive_rl

competitive_rl.register_competitive_envs()
log_dir = "./tmp/"
model = PPO2.load(log_dir + "ppo_pong.zip")

# Load the saved statistics
env = DummyVecEnv([lambda: gym.make("cPong-v0")])
#env = VecNormalize.load(stats_path, env)
#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()