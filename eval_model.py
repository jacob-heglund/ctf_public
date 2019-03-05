######################
## program controls
load_ckpt_dir = 'b1_r0_m20_s150--2019-03-04--132150.003907'
load_episode = 3000

# record_video = 0: eval over eval_episodes
# record_video = 1: record a number of episodes (renders for each episode, so takes a long time!)
# plays a small amount of games until
record_video = 1

# this part only applicable if record_video = 0
eval_episodes = 1000 # choose nice, even numbers please
window = int(eval_episodes / 25)

# this part only applicable if record_video = 1
# set the number of videos for successful and failed runs that we want to record
num_success = 5
num_failure = 5 # failed episodes tend to be much longer than successful ones


#TODO record agent actions and q-values during evaluation
#TODO visualize actions at each timestep and the associated q-values (https://www.youtube.com/watch?v=XjsY8-P4WHM)
#TODO put a frame counter and episode number as part of the legend on each game

#TODO make eval work with standard training functions if possible, those functions will all be put in a file for easier implementation of different algorithms
######################
## regular imports
import sys
import argparse
import os
import gym
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder

import moviepy.editor as mp
from moviepy.video.fx.all import speedx
import numpy as np
from numpy import shape
import matplotlib.pyplot as plt
import pandas as pd
import math
import datetime
import time
from collections import deque
import random
import json

## Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

## custom modules
import policy.random_actions
import gym_cap
from utility.dataModule import one_hot_encoder
from utility.utils import MovingAverage as MA
from utility.utils import Experience_buffer, discount_rewards
from utility.utilityRL import DQN, ReplayBuffer

######################
# file / data management
def save_data(episode, step_list, reward_list, loss_list, epsilon_list):
    '''
    Saves episode data and makes a plot for visualizing training.
    Args:
        episode (int): Current episode
        step_list (list): Contains the length of each episode in frames
        reward_list (list): Contains the reward for each episode
        loss_list (list): Contains the loss for each episode
        epsilon_list (list): Contains the exploration rate for each episode
    '''

    # - save episode data
    # - make a plot
    # save evaluation data
    step_list = np.asarray(step_list)
    reward_list = np.asarray(reward_list)
    loss_list = np.asarray(loss_list)
    epsilon_list = np.asarray(epsilon_list)
    episode_save = np.vstack((step_list, reward_list, loss_list, epsilon_list))

    # get averages
    step_avg = np.mean(step_list)
    step_list_avg = step_avg*np.ones(np.shape(step_list))
    reward_avg = np.mean(reward_list)
    reward_list_avg = reward_avg*np.ones(np.shape(reward_list))
    success = np.count_nonzero(reward_list == 100)
    percent_success = round((success / eval_episodes)*100, 3)

    fn = 'episode_data.txt'
    fp = os.path.join(data_dir, fn)
    with open(fp, 'w') as f:
        np.savetxt(f, episode_save)

    plt.figure(figsize = [10,8])
    plt.subplot(211)
    plt.plot(pd.Series(step_list).rolling(window).mean(), label = 'Length (frames)')
    plt.plot(step_list_avg, label = 'Mean Episode Length = {}'.format(round(step_avg, 1)), linewidth = .7)
    plt.title('Frames per Episode (Moving Average {}-episode Window)'.format(window))
    plt.ylabel('Frames')
    plt.xlabel('Episode')
    plt.legend(loc = 'upper right')

    plt.subplot(212)
    plt.plot(pd.Series(reward_list).rolling(window).mean(), label = 'Reward')
    plt.plot(reward_list_avg, label = 'Mean Reward = {}'.format(round(reward_avg, 1)), linewidth = .7)
    plt.title('Reward per Episode (Moving Average, {}-episode Window)'.format(window))
    plt.ylabel('Reward')


    success_str = 'Succesful Eval. Episodes: {}/{} ({}%)\n'.format(success, eval_episodes, percent_success)

    training_string = 'Network: DQN\nTraining Episodes: {}\n--------------------------------------------------\n'.format(load_episode)

    num_UGV_red, num_UAV_red = count_team_units(env.get_team_red)
    num_UGV_blue, num_UAV_blue = count_team_units(env.get_team_blue)
    env_param_string = 'Map Size: {}\nMax # Frames per Episode: {}\nVision Radius: {}\n# Blue UGVs: {}\n# Blue UAVs: {}\n# Red UGVs: {}\n# Red UAVs: {}'.format(train_params['map_size'], train_params['max_episode_length'], train_params['vision_radius'], num_UGV_blue, num_UAV_blue,num_UGV_red, num_UAV_red)

    text = success_str + training_string + env_param_string
    bbox_props = dict(boxstyle='square', fc = 'white')
    plt.xlabel(text, bbox = bbox_props)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.legend(loc = 'upper right')

    fn = 'eval_data.png'
    fp = os.path.join(data_dir, fn)
    plt.savefig(fp, dpi=300)
    plt.close()

def load_model(load_episode):
    '''
    Loads a model from a particular training episode for evaluation.
    Args:
        load_episode (int): Saved training episode to be evaluated

    Returns:
        online_model (pytorch model): loaded model from the training episode
    '''

    #TODO make this generalize to multiple network architectures, save network type in train_params
    online_model = DQN(num_states, num_actions, train_params['batch_size'])

    # load only the state dict
    fn = 'episode_' + str(load_episode) + '.model'
    fp = os.path.join(ckpt_dir, fn)
    load_model = torch.load(fp)
    online_model.load_state_dict(load_model.state_dict())
    online_model = online_model.to(device)
    return online_model

def setup_data_storage(load_episode):
    '''
    Inits directories and training data lists to be used during evaluation.

    Args:
        load_episode (int): Saved training episode to be evaluated

    Returns:
        ckpt_dir (str): directory for loading the model
        data_dir (str): directory for saving evaluation data
        train_params (dict): dict of hyperparameters used during training and evaluation
        frame_count (list): number of frames that have passed
        step_list (list): Contains the length of each episode in frames
        reward_list (list): Contains the reward for each episode
        loss_list (list): Contains the loss for each episode
        epsilon_list (list): Contains the exploration rate for each episode
    '''

    # set checkpoint save directory
    ckpt_dir = './data/' + load_ckpt_dir

    eval_dir = os.path.join(ckpt_dir, 'eval')
    dir_exist = os.path.exists(eval_dir)
    if not dir_exist:
        os.mkdir(eval_dir)

    data_dir =  os.path.join(eval_dir, 'episode_' + str(load_episode))
    dir_exist = os.path.exists(data_dir)
    if not dir_exist:
        os.mkdir(data_dir)

    # setup hyperparameters
    #TODO right now, this will load the map parameters used during training
    # if you want to evaluate on a different sized map, change it here
    fn = 'train_params.json'
    fp = os.path.join(ckpt_dir, fn)
    with open(fp, 'r') as f:
        train_params = json.load(f)

    # init lists for training data
    step_list = []
    reward_list = []
    loss_list = []
    epsilon_list = []

    # init frame_count
    frame_count = 0

    return ckpt_dir, data_dir, train_params, frame_count, step_list, reward_list, loss_list, epsilon_list

######################
# RL functions

def count_team_units(team_list):
    '''
    Counts total UAVs and UGVs for a team.

    Args:
        team_list (list): list of team members.  Use env.get_team_(red or blue) as input.

    Returns:
        num_UGV (int): number of ground vehicles
        num_UAV (int): number of aerial vehicles
    '''

    num_UAV = 0
    num_UGV = 0
    for i in range(len(team_list)):
        if isinstance(team_list[i], gym_cap.envs.agent.GroundVehicle):
            num_UGV += 1
        elif isinstance(team_list[i], gym_cap.envs.agent.AerialVehicle):
            num_UAV += 1
        else:
            continue
    return num_UGV, num_UAV

def format_state_for_action(state):
    '''
    Formats the raw input state for generating actions.

    Args:
        state (numpy array): Has shape (num_agents, map_x, map_y, num_channels)

    Returns:
        s (torch tensor): Has shape (num_agents, num_channels, map_x, map_y)
    '''

    s = np.swapaxes(state, 3, 2)
    s = np.swapaxes(s, 2, 1)

    s = torch.from_numpy(s).type(torch.FloatTensor).to(device).unsqueeze(0)

    return s

def get_action(state, epsilon, team_list):
    '''
    Generates actions for a single team of agents for a single timestep of the sim.

    Args:
        state (np array): Raw input state from the CTF env
        epsilon (float): Probability of taking a random action
        team_list (list): list of team members.  Use env.get_team_(red or blue) as input.

    Returns:
        action_list (list): List of actions for each agent to take
    '''

    if np.random.rand(1) < epsilon:
        action_list = random.choices(action_space, k = num_units)

    else:
        action_list = []
        state = format_state_for_action(state)
        with torch.no_grad():
            for i in range(num_units):
                q_values = online_model.forward(state[:, i, :, :, :])
                _, action = torch.max(q_values, 1)
                action_list.append(int(action.data))
                #TODO for optimized version
                # action_list = list(action.numpy().astype(int))

    return action_list

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

    if record_video:
        video_dir = os.path.join(data_dir, 'raw_videos')

        dir_exist = os.path.exists(video_dir)
        if not dir_exist:
            os.mkdir(video_dir)

        video_fn = 'episode_' + str(episode) + '.mp4'
        video_path = os.path.join(video_dir, video_fn)

        video_recorder = VideoRecorder(env, video_path, enabled = record_video)

    env.reset(map_size = train_params['map_size'], policy_red = policy_red)

    episode_length = 0.
    episode_loss = 0.
    done = 0
    #TODO success_flag doesn't work as intended, some videos classified as successful are not successes
    success_flag = -1

    while (done == 0):
        # set exploration rate for this frame
        epsilon = 0
        if record_video:
            video_recorder.capture_frame()

        # state consists of the centered observations of each agent
        state = one_hot_encoder(env.unwrapped._env, env.unwrapped.get_team_blue, vision_radius = train_params['vision_radius'])

        # action is a list containing the actions for each agent
        action = get_action(state, epsilon, env.get_team_blue)

        _ , reward, done, _ = env.unwrapped.step(entities_action = action)

        episode_length += 1
        frame_count += 1

        # check for success or failure from regular gameplay
        if (done == 1):
            if (reward > 0):
                success_flag = 1

            if (reward == 0):
                success_flag = 0

        # give a failure if the episode goes on too long
        if (episode_length >= train_params['max_episode_length']):
            reward = 0.0
            success_flag = 0
            done = True

        # end the episode
        if done:
            if record_video:
                video_recorder.capture_frame()
                video_recorder.close()
                vid = mp.VideoFileClip(video_path)

                if (success_flag == 1):
                    vid_success.append(vid)

                elif (success_flag == 0):
                    vid_failure.append(vid)
                else:
                    print('Success Flag Error')
                    continue
            return episode_loss, episode_length, reward, epsilon

######################
## setup for training
# init environment
env_id = 'cap-v0'
env = gym.make(env_id)
num_UGV_red, num_UAV_red = count_team_units(env.get_team_red)
num_UGV_blue, num_UAV_blue = count_team_units(env.get_team_blue)
num_units = num_UGV_blue + num_UAV_blue
print('Blue UGVs: {}\nBlue UAVs: {}\nRed UGVs: {}\nRed UAVs: {}'.format(num_UGV_blue, num_UAV_blue, num_UGV_red, num_UAV_red))
#TODO print map size too

# storage for training data
ckpt_dir, data_dir, train_params, frame_count, step_list, reward_list, loss_list, epsilon_list = setup_data_storage(load_episode)

policy_red = policy.random_actions.PolicyGen(env.unwrapped.get_map, env.unwrapped.get_team_red)
env.reset(map_size = train_params['map_size'], policy_red = policy_red)

# get fully observable state
num_states = (2*train_params['vision_radius'] + 1)**2
action_space = [0, 1, 2, 3, 4]
num_actions = len(action_space)

# setup neural net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
online_model = load_model(load_episode)

######################
if __name__ == '__main__':
    vid_success = []
    vid_failure = []

    time1 = time.time()
    if record_video:
            episode = 0
            done_flag = 0

            while (not done_flag):
                loss, length, reward, epsilon = play_episode()

                # save episode data after the episode is done
                step_list.append(length)
                loss_list.append(loss / length)
                reward_list.append(reward)
                epsilon_list.append(epsilon)

                print('Success: {}/{} ---- Failure: {}/{} ---- Episode: {} ---- Runtime: {} '.format(len(vid_success), num_success, len(vid_failure), num_failure, episode, round(time.time()-time1, 3)))

                if (len(vid_success) >= num_success) and (len(vid_failure) >= num_failure):
                    done_flag = 1


                episode += 1

    else:
        for episode in range(eval_episodes):
            loss, length, reward, epsilon = play_episode()

            # save episode data after the episode is done
            step_list.append(length)
            loss_list.append(loss / length)
            reward_list.append(reward)
            epsilon_list.append(epsilon)

            if episode % (eval_episodes/20) == 0:
                print('Episode: {}/{} ({}) ---- Runtime: {} '.format(episode, eval_episodes, round(float(episode) / float(eval_episodes), 3), round(time.time()-time1, 3)))

if not record_video:
    save_data(episode, step_list, reward_list, loss_list, epsilon_list)

env.unwrapped.close()

if record_video:
    if num_success > 0:
        vid_success = vid_success[0:num_success]

        vid = mp.concatenate_videoclips(vid_success)
        vid = speedx(vid, 0.1)
        legend = mp.ImageClip('./images/legend.png', duration = vid.duration)

        final_vid = mp.clips_array([[legend, vid]])
        fp = os.path.join(data_dir, 'success.mp4')
        final_vid.write_videofile(fp)

    if num_failure > 0:
        vid_failure = vid_failure[0:num_failure]

        vid = mp.concatenate_videoclips(vid_failure)
        vid = speedx(vid, 0.25)
        legend = mp.ImageClip('./images/legend.png', duration = vid.duration)

        final_vid = mp.clips_array([[legend, vid]])
        fp = os.path.join(data_dir, 'failure.mp4')
        final_vid.write_videofile(fp)


