######################
## program controls
load_dir = '' # set as empty string if not loading a checkpoint
load_episode = 3000 # set to 0 if not loading a checkpoint

render = 0
# training picks up at load_episode and runs until total_episodes
total_episodes = 3000

if load_episode > total_episodes:
    print('Please set load_episode such that load_episode < total_episodes')
    raise ValueError()

# training hyperparameters
def setup_hyperparameters():
    '''
    Init all relevant training and evaluation hyperparameters.

    Returns:
        train_params (dict): contains all the training hyperparameters
    '''

    train_params = {}
    ## game hyperparameters
    train_params['num_episodes'] = total_episodes
    # train_params['vision_radius'] = 4
    train_params['vision_radius'] =11

    # train_params['map_size'] = 5
    # train_params['max_episode_length'] = 15

    # train_params['map_size'] = 8
    # train_params['max_episode_length'] = 100

    # train_params['map_size'] = 10
    # train_params['max_episode_length'] = 100

    train_params['map_size'] = 20
    train_params['max_episode_length'] = 150

    ## training hyperparameters
    train_params['epsilon_start'] = 1.0
    train_params['epsilon_final'] = 0.02
    train_params['epsilon_decay'] = 150*5000 # play 5000 games with a 'high' chance of random action for better exploration
    train_params['gamma'] = 0.99 # future reward discount
    train_params['learning_rate'] = 10**-4
    train_params['batch_size'] = 50 # number of transitions to sample from replay buffer

    train_params['replay_buffer_capacity'] = 2000
    train_params['replay_buffer_init'] = 100 # number of frames to simulate before we start sampling from the buffer
    train_params['train_online_model_frame'] = 4 # number of frames between training the online network

    return train_params

######################
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

## Pytorch
import torch
import torch.nn as nn
import torch.optim as optim

## custom modules
import policy.random_actions
import gym_cap
#TODO upgrade to improved one_hot after multi agent is working
# from utility.dataModule import one_hot_encoder_v2 as one_hot_encoder
from utility.dataModule import one_hot_encoder
from utility.utils import MovingAverage as MA
from utility.utils import Experience_buffer, discount_rewards
from utility.utilityRL import DQN, ReplayBuffer

######################
# file / data management
def save_data(episode, step_list, reward_list, loss_list, epsilon_list):
    '''
    Saves model weights, hyperparameters, episode data, and makes a plot for visualizing training.
    Args:
        episode (int): Current episode
        step_list (list): Contains the length of each episode in frames
        reward_list (list): Contains the reward for each episode
        loss_list (list): Contains the loss for each episode
        epsilon_list (list): Contains the exploration rate for each episode
    '''

    # save weights
    fn = 'episode_' + str(episode) + '.model'
    fp = os.path.join(ckpt_dir, fn)
    torch.save(online_model, fp)

    # save hyperparameters
    fn = 'train_params.json'
    fp = os.path.join(ckpt_dir, fn)

    if episode == 0:
        with open(fp, 'w') as f:
            json.dump(train_params, f)

    # save training data
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

    window = 100
    fn = 'episode_data.txt'
    fp = os.path.join(ckpt_dir, fn)
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
    plt.legend(loc = 'upper right')

    num_UGV_red, num_UAV_red = count_team_units(env.get_team_red)
    num_UGV_blue, num_UAV_blue = count_team_units(env.get_team_blue)

    text = 'Map Size: {}\nMax # Frames per Episode: {}\nVision Radius: {}\n# Blue UGVs: {}\n# Blue UAVs: {}\n# Red UGVs: {}\n# Red UAVs: {}'.format(train_params['map_size'], train_params['max_episode_length'], train_params['vision_radius'], num_UGV_blue, num_UAV_blue,num_UGV_red, num_UAV_red)

    bbox_props = dict(boxstyle='square', fc = 'white')
    plt.xlabel(text, bbox = bbox_props)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fn = 'training_data.png'
    fp = os.path.join(ckpt_dir, fn)
    plt.savefig(fp, dpi=300)
    plt.close()

def load_model(load_episode):
    '''
    Loads a model to continue training, or loads a new model for a new training run.

    Args:
        load_episode (int): Saved training episode for continuing a training run

    Returns:
        online_model (pytorch model): loaded model from the training episode
    '''

    if (load_dir == ''):
        #TODO make this generalize to multiple network architectures, save network type in train_params
        online_model = DQN(num_obsv_states, num_actions, train_params['batch_size'])
        online_model = online_model.to(device)

    else:
        #TODO make this generalize to multiple network architectures
        online_model = DQN(num_obsv_states, num_actions, train_params['batch_size'])

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
        ckpt_dir (str): directory for saving training data
        train_params (dict): dict of hyperparameters used during training and evaluation
        frame_count (list): number of frames that have passed
        step_list (list): Contains the length of each episode in frames
        reward_list (list): Contains the reward for each episode
        loss_list (list): Contains the loss for each episode
        epsilon_list (list): Contains the exploration rate for each episode
    '''

    if (load_dir == ''):
        # setup hyperparameters
        train_params = setup_hyperparameters()

        # set checkpoint save directory
        time = datetime.datetime.now()

        num_blue_UGV, _ = count_team_units(env.get_team_blue)
        num_red_UGV, _ = count_team_units(env.get_team_red)
        map_size = train_params['map_size']
        step = train_params['max_episode_length']
        map_params_str = 'b' + str(num_blue_UGV) + '_r' + str(num_red_UGV) + '_m' + str(map_size) + '_s' + str(step) + '--'
        time_str = str(time).replace(' ', '--').replace(':', '')

        ckpt_dir = './data/' + map_params_str + time_str
        dir_exist = os.path.exists(ckpt_dir)
        if not dir_exist:
            os.mkdir(ckpt_dir)

        # init frame count
        frame_count = 0

        # init lists for training data
        step_list = []
        reward_list = []
        loss_list = []
        epsilon_list = []

    else:
        # set checkpoint save directory
        ckpt_dir = './data/' + load_dir

        # setup hyperparameters
        fn = 'train_params.json'
        fp = os.path.join(ckpt_dir, fn)
        with open(fp, 'r') as f:
            train_params = json.load(f)

        train_params['num_episodes'] = total_episodes

        # init lists for training data
        fn = 'episode_data.txt'
        fp = os.path.join(ckpt_dir, fn)
        with open(fp, 'r') as f:
            data = np.loadtxt(f)

        step_list = np.ndarray.tolist(data[0, 0:load_episode])
        reward_list = np.ndarray.tolist(data[1, 0:load_episode])
        loss_list = np.ndarray.tolist(data[2, 0:load_episode])
        epsilon_list = np.ndarray.tolist(data[3, 0:load_episode])

        # init frame_count
        frame_count = np.sum(step_list)

    return ckpt_dir, train_params, frame_count, step_list, reward_list, loss_list, epsilon_list

######################
# RL functions
#TODO have get_action, epsilon_by_frame, and train_online_model as functions in the RL algorithm class
#NOTE this cannot be easily done. for pytorch to save the model as a model object, the DQN class can only have init and forward functions
#TODO workaround: have a separate module for each algorithm

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

def epsilon_by_frame(frame_count):
    '''
    Generates random action probability based on the number of total frames that have passed.

    Args:
        frame_count (int): Number of total simulation frames that have occurred.

    Returns:
        epsilon_curr (float): Probability of taking a random action.
    '''

    epsilon_curr = train_params['epsilon_final'] + (train_params['epsilon_start'] - train_params['epsilon_final']) * math.exp(-1. * frame_count / train_params['epsilon_decay'])

    return epsilon_curr

def train_online_model(batch_size):
    '''
    Trains the model based on the Q-Learning algorithm with Experience Replay.

    Args:
        batch_size (int): Number of transitions to be sampled from the replay buffer.

    Returns:
        loss (float): loss for the batch of training
    '''

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    # evaluate for each agent separately
    #TODO optimize so the state only goes through the network 1 time
    for i in range(num_units):

        # get all the q-values for actions at state and next state
        q_values = online_model.forward(state[:, i, :, :, :])
        next_q_values = online_model.forward(next_state[:, i, :, :, :])

        # find Q(s, a) for the action taken during the sampled transition
        agent_action = action[:, i].unsqueeze(1)
        state_action_value = q_values.gather(1, agent_action).squeeze(1)

        # Find Q(s_next, a_next) for an optimal agent (take the action with max q-value in s_next)
        next_state_action_value = next_q_values.max(1)[0]

        # if done, multiply next_state_action_value by 0, else multiply by 1
        one = np.ones(batch_size)
        done_mask = np.array(one-done).astype(int)
        done_mask = torch.from_numpy(done_mask).type(torch.FloatTensor)

        discounted_next_value = (train_params['gamma'] * next_state_action_value)
        discounted_next_value = discounted_next_value.type(torch.FloatTensor)

        # Compute the target of current q-values
        target_value = reward + discounted_next_value * done_mask

        loss = criterion(state_action_value, target_value)

        online_model.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

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
        epsilon = epsilon_by_frame(frame_count)

        # state consists of the centered observations of each agent
        state = one_hot_encoder(env._env, env.get_team_blue, vision_radius = train_params['vision_radius'])

        # action is a list containing the actions for each agent
        action = get_action(state, epsilon, env.get_team_blue)

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
            if (frame_count % train_params['train_online_model_frame']) == 0:
                loss = train_online_model(train_params['batch_size'])
                episode_loss += loss

        # end the episode
        if done:
            return episode_loss, episode_length, reward, epsilon

######################
## setup for training
# init environment
env_id = 'cap-v0'
env = gym.make(env_id)

num_UGV_red, num_UAV_red = count_team_units(env.get_team_red)
num_UGV_blue, num_UAV_blue = count_team_units(env.get_team_blue)
num_units = num_UGV_blue + num_UAV_blue

# storage for training data
ckpt_dir, train_params, frame_count, step_list, reward_list, loss_list, epsilon_list = setup_data_storage(load_episode)
map_str = 'Map Size: {}x{}\nMax Frames per Episode: {}\n'.format(train_params['map_size'], train_params['map_size'], train_params['max_episode_length'])
agent_str = 'Blue UGVs: {}\nBlue UAVs: {}\nRed UGVs: {}\nRed UAVs: {}'.format(num_UGV_blue, num_UAV_blue, num_UGV_red, num_UAV_red)
print(map_str + agent_str)

# policy_red = policy.roomba.PolicyGen(env.get_map, env.get_team_red)
policy_red = policy.stay_still.PolicyGen(env.get_map, env.get_team_red)
# policy_red = policy.random_actions.PolicyGen(env.get_map, env.get_team_red)
env.reset(map_size = train_params['map_size'], policy_red = policy_red)

# init replay buffer
replay_buffer = ReplayBuffer(train_params['replay_buffer_capacity'])

# get fully observable state
num_obsv_states = (train_params['vision_radius']*2 + 1)**2
action_space = [0, 1, 2, 3, 4]
num_actions = len(action_space)


# setup neural net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
online_model = load_model(load_episode)
criterion = nn.MSELoss()
optimizer = optim.Adam(online_model.parameters(), lr = train_params['learning_rate'])

######################
if __name__ == '__main__':
    time1 = time.time()
    for episode in range(load_episode, train_params['num_episodes']+1):
        loss, length, reward, epsilon = play_episode()

        # save episode data after the episode is done
        step_list.append(length)
        loss_list.append(loss / length)
        reward_list.append(reward)
        epsilon_list.append(epsilon)

        if episode % 10 == 0:
            print('Episode: {}/{} ({}) ---- Runtime: {} '.format(episode, train_params['num_episodes'], round(float(episode) / float(train_params['num_episodes']), 3), round(time.time()-time1, 3)))

        if episode % 250 == 0:
            save_data(episode, step_list, reward_list, loss_list, epsilon_list)
