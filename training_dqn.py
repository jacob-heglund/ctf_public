#################################
## program controls
load_dir = '' # set as empty string if not loading a checkpoint
load_episode = 0 # set to 0 if not loading a checkpoint

render = 0
# training picks up at load_episode and runs until total_episodes
total_episodes = 10000

if load_episode > total_episodes:
    print('Please set load_episode such that load_episode < total_episodes')
    raise ValueError()

#################################
## regular imports
import sys
import argparse
import os
import gym
import numpy as np
from numpy import shape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import math
import datetime
import time
from collections import deque
import random
import json
import copy

## Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

## custom modules
import gym_cap
from utility.utility_env import one_hot_encoder
from utility.utility_DQN import DQN, ReplayBuffer, epsilon_by_frame, get_action, train_model, cnn_output_size, save_data, load_model, set_up_data_storage, count_team_units

import policy
#################################
def play_episode():
    '''
    Plays a single episode of the sim

    Returns:
        episode_loss (float): TODO: how to get a good measure of loss with 4 agents?
        episode_length (int): number of frames in the episode
        reward (int): final reward for the blue team
        epsilon (float): Probability of taking a random action
    '''

    global frame_count
    env.reset(map_size = train_params['map_size'], policy_red = policy_red)

    episode_length = 0.
    episode_loss = 0.
    done = 0

    while (done == 0):
        if render:
            env.render()

        # set exploration rate for this frame
        epsilon = epsilon_by_frame(frame_count, train_params)

        # state consists of the centered observations for each agent
        # np.shape(state) = (num_agents, vision_radius + 1, vision_radius + 1, num_channels)
        state = one_hot_encoder(env._env, env.get_team_blue, vision_radius = train_params['vision_radius'])

        # action is a list containing the actions for each agent
        action = get_action(online_model, state, epsilon, env.get_team_blue)

        _ , reward, done, _ = env.step(entities_action = action)

        # reward = reward / 100.
        next_state = one_hot_encoder(env._env, env.get_team_blue, vision_radius = train_params['vision_radius'])
        episode_length += 1
        frame_count += 1

        # set Done flag if episode goes for too long without reaching the flag
        if episode_length >= train_params['max_episode_length']:
            reward = 0.0
            done = True

        # store the transition in replay buffer
        replay_buffer.push(state, action, reward, next_state, done)

        # train the network
        if len(replay_buffer) > train_params['replay_buffer_init']:
            if (frame_count % train_params['train_model_frame']) == 0:
                loss = train_model(online_model, replay_buffer, env.get_team_blue, train_params, optimizer, loss_function)
                episode_loss += loss

        # end the episode
        if done:
            return episode_loss, episode_length, reward, epsilon

#################################
## set up training
# init environment
env_id = 'cap-v0'
env = gym.make(env_id)

num_UGV_red, num_UAV_red = count_team_units(env.get_team_red)
num_UGV_blue, num_UAV_blue = count_team_units(env.get_team_blue)
num_units_blue = num_UGV_blue + num_UAV_blue

# set up storage for training data, print relevant data
ckpt_dir, train_params, frame_count, step_list, reward_list, loss_list, epsilon_list = set_up_data_storage(load_episode, load_dir, total_episodes, env)

map_str = \
'Map Size: {}x{}\n\
Max Frames per Episode: {}\n'\
.format(train_params['map_size'], train_params['map_size'], train_params['max_episode_length'])

agent_str = \
'Blue UGVs: {}\n\
Blue UAVs: {}\n\
Red UGVs: {}\n\
Red UAVs: {}'\
.format(num_UGV_blue, num_UAV_blue, num_UGV_red, num_UAV_red)
print(map_str + agent_str)

# init replay buffer
replay_buffer = ReplayBuffer(train_params['replay_buffer_capacity'])

# set up neural net
online_model = load_model(load_episode, load_dir, train_params)
optimizer = optim.Adam(online_model.parameters(), lr = train_params['learning_rate'])
loss_function = nn.MSELoss()

def update_red_policy(online_model, train_params):
    if train_params['enemy_policy'] == 'roomba':
        return policy.roomba.PolicyGen(env.get_map, env.get_team_red)

    elif train_params['enemy_policy'] == 'stay_still':
        return policy.stay_still.PolicyGen(env.get_map, env.get_team_red)

    elif train_params['enemy_policy'] == 'random':
        return policy.random_actions.PolicyGen(env.get_map, env.get_team_red)

    elif train_params['enemy_policy'] == 'self_play_dqn':
        return policy.self_play_dqn.PolicyGen(online_model, train_params, env.get_map, env.get_team_red, env)

    else:
        print('Invalid enemy policy, stopping program execution')
        sys.exit()

policy_red = update_red_policy(online_model, train_params)
print('Red Policy: ' + train_params['enemy_policy'])

env.reset(map_size = train_params['map_size'], policy_red = policy_red)

#################################
if __name__ == '__main__':
    time1 = time.time()
    for episode in range(load_episode, train_params['num_episodes']+1):

        # update enemy policy
        if (episode % 100 == 0) and (train_params['enemy_policy'] == 'self_play_dqn'):
            policy_red = update_red_policy(online_model, train_params)

        loss, length, reward, epsilon = play_episode()

        # save episode data after the episode is done
        step_list.append(length)
        loss_list.append(loss / length)
        reward_list.append(reward)
        epsilon_list.append(epsilon)

        if episode % 10 == 0:
            print('Episode: {}/{} ({}) ---- Runtime: {} '.format(episode, train_params['num_episodes'], round(float(episode) / float(train_params['num_episodes']), 3), round(time.time()-time1, 3)))

        if episode % 100 == 0:
            save_data(online_model, episode, step_list, reward_list, loss_list, epsilon_list, ckpt_dir, train_params, env)
