import math

import gym
import pygame
from gym.envs.registration import register

from competitive_rl.car_racing.car_racing_multi_players import CarRacing
from competitive_rl.car_racing.controller import key_phrase


def register_competitive_envs():
    register(
        id="cCarRacing-v0",
        entry_point=CarRacing,
        max_episode_steps=1000,
        reward_threshold=900
    )
    print(
        "Register car_racing_multiple_players environments.")

def log(x):
    return math.log(x,2)

def e(x):
    return x * log(x)


if __name__ == "__main__":
    num_player = 2
    register_competitive_envs()
    #env = CarRacing(num_player=num_player)
    env = gym.make('cCarRacing-v0')
    # example: env.reset(use_local_track="./track/test.json",record_track_to="")
    # example: env.reset(use_local_track="",record_track_to="./track")
    # env.reset(use_local_track="./track/test3.json",record_track_to="")
    env.reset(use_local_track="", record_track_to="")
    a = [[0.0, 0.0, 0.0] for _ in range(num_player)]
    print(env.seed())
    clock = pygame.time.Clock()
    while True:
        env.manage_input(key_phrase(a))
        if env.isrender:
            env.render()
        observation, reward, done, info = env.step(a)

        if env.show_all_car_obs:
            env.show_all_obs([observation], grayscale=True)
        clock.tick(60)
        #fps = clock.get_fps()
        #print(fps)