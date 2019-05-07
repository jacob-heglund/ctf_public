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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

###############################
## RL functions
def set_up_hyperparameters(total_episodes):
    '''
    Init all relevant training and evaluation hyperparameters.

    Returns:
        train_params (dict): contains all the training hyperparameters
    '''

    train_params = {}
    ## game hyperparameters
    train_params['num_episodes'] = total_episodes
    train_params['vision_radius'] = 20

    # train_params['map_size'] = 5
    # train_params['max_episode_length'] = 15

    # train_params['map_size'] = 8
    # train_params['max_episode_length'] = 100

    # train_params['map_size'] = 10
    # train_params['max_episode_length'] = 100

    train_params['map_size'] = 20
    train_params['max_episode_length'] = 150

    # train_params['map_size'] = 50
    # train_params['max_episode_length'] = 150

    ## training hyperparameters
    train_params['network_architecture'] = 'DQN'

    # currently supported: roomba, random, stay_still, self_play_dqn
    train_params['enemy_policy'] = 'self_play_dqn'

    train_params['epsilon_start'] = 1.0
    train_params['epsilon_final'] = 0.02
    train_params['epsilon_decay'] = 150*5000 # play 5000 games with a 'high' chance of random action for better exploration
    train_params['gamma'] = 0.99 # future reward discount
    train_params['learning_rate'] = 10**-4
    train_params['batch_size'] = 50 # number of transitions to sample from replay buffer

    train_params['replay_buffer_capacity'] = 2000
    train_params['replay_buffer_init'] = 100 # number of frames to simulate before we start sampling from the buffer
    train_params['train_model_frame'] = 4 # number of frames between training the online network

    return train_params

def cnn_output_size(w, k, p, s):
    return ((w-k+2*p)/s)+1

def epsilon_by_frame(frame_count, train_params):
    '''
    Generates random action probability based on the number of total frames that have passed.

    Args:
        frame_count (int): Number of total simulation frames that have occurred.

    Returns:
        epsilon_curr (float): Probability of taking a random action.
    '''

    epsilon_curr = train_params['epsilon_final'] + (train_params['epsilon_start'] - train_params['epsilon_final']) * math.exp(-1. * frame_count / train_params['epsilon_decay'])

    return epsilon_curr

def format_state_for_action(state):
    '''
    Formats the one-hot input state for generating actions

    Args:
        state (numpy array): Has shape (num_agents, map_x, map_y, num_channels)

    Returns:
        s (torch tensor): Has shape (num_agents, num_channels, map_x, map_y)
    '''

    s = np.swapaxes(state, 3, 2)
    s = np.swapaxes(s, 2, 1)

    s = torch.from_numpy(s).type(torch.FloatTensor).to(device).unsqueeze(0)

    return s

def get_action(model, state, epsilon, team_list):
    '''
    Generates actions for a single team of agents for a single timestep of the sim.

    Args:
        state (np array): Raw input state from the CTF env
        epsilon (float): Probability of taking a random action
        team_list (list): list of team members.  Use env.get_team_(red or blue) as input.

    Returns:
        action_list (list): List of actions for each agent to take
    '''
    action_space = [0, 1, 2, 3, 4]

    if np.random.rand(1) < epsilon:
        action_list = random.choices(action_space, k = len(team_list))

    else:
        action_list = []
        #TODO wouldn't this only work for 1 team member at a time?
        state = format_state_for_action(state)
        with torch.no_grad():
            for i in range(len(team_list)):
                q_values = model.forward(state[:, i, :, :, :])
                _, action = torch.max(q_values, 1)
                action_list.append(int(action.data))
                #TODO for optimized version
                # action_list = list(action.numpy().astype(int))

    return action_list

def train_model(model, replay_buffer, team_list, train_params, optimizer, loss_function):
    '''
    Trains the model based on the Q-Learning algorithm with Experience Replay.

    Args:
        batch_size (int): Number of transitions to be sampled from the replay buffer.

    Returns:
        loss (float): loss for the batch of training
    '''

    state, action, reward, next_state, done = replay_buffer.sample(train_params['batch_size'])

    # evaluate for each agent separately
    #TODO optimize so the state only goes through the network 1 time
    for i in range(len(team_list)):

        # get all the q-values for actions at state and next state
        q_values = model.forward(state[:, i, :, :, :])
        next_q_values = model.forward(next_state[:, i, :, :, :])

        # find Q(s, a) for the action taken during the sampled transition
        agent_action = action[:, i].unsqueeze(1)
        state_action_value = q_values.gather(1, agent_action).squeeze(1)

        # Find Q(s_next, a_next) for an optimal agent (take the action with max q-value in s_next)
        next_state_action_value = next_q_values.max(1)[0]

        # if done, multiply next_state_action_value by 0, else multiply by 1
        one = np.ones(train_params['batch_size'])
        done_mask = np.array(one-done).astype(int)
        done_mask = torch.from_numpy(done_mask).type(torch.FloatTensor)

        discounted_next_value = (train_params['gamma'] * next_state_action_value)
        discounted_next_value = discounted_next_value.type(torch.FloatTensor)

        # Compute the target of current q-values
        target_value = reward + discounted_next_value * done_mask

        loss = loss_function(state_action_value, target_value)
        # criterion(state_action_value, target_value)

        model.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

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

###############################
## file and data management
def save_data(model, episode, step_list, reward_list, loss_list, epsilon_list, ckpt_dir, train_params, env):
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
    torch.save(model, fp)

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

def load_model(load_episode, load_dir, train_params):
    '''
    Loads a model to continue training, or loads a new model for a new training run.

    Args:
        load_episode (int): Saved training episode for continuing a training run

    Returns:
        online_model (pytorch model): loaded model from the training episode
    '''

    # get size of observable state
    num_obsv_states = (train_params['vision_radius']*2 + 1)**2
    action_space = [0, 1, 2, 3, 4]
    num_actions = len(action_space)

    if (load_dir == ''):
        model = DQN(num_obsv_states, num_actions, train_params['batch_size'])
        model = model.to(device)

    else:
        model = DQN(num_obsv_states, num_actions, train_params['batch_size'])

        # load state dict
        fn = 'episode_' + str(load_episode) + '.model'
        fp = os.path.join(ckpt_dir, fn)
        temp_load_model = torch.load(fp)
        model.load_state_dict(temp_load_model.state_dict())
        model = model.to(device)
    return model

def set_up_data_storage(load_episode, load_dir, total_episodes, env):
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
        # set up hyperparameters
        train_params = set_up_hyperparameters(total_episodes)

        # set checkpoint save directory
        num_blue_UGV, _ = count_team_units(env.get_team_blue)
        num_red_UGV, _ = count_team_units(env.get_team_red)
        map_params_str = \
        train_params['enemy_policy'] + '_' + \
        train_params['network_architecture'] + '_' + \
        'b' + str(num_blue_UGV) + \
        '_r' + str(num_red_UGV) + \
        '_m' + str(train_params['map_size']) + \
        '_s' + str(train_params['max_episode_length']) + '--'

        time_str = str(datetime.datetime.now()).replace(' ', '--').replace(':', '')

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

        # set up hyperparameters
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

###############################
## network architectures
class DQN(nn.Module):
    def __init__(self, num_obsv_states, num_actions, batch_size):
        '''
        Pytorch neural network class for value-function approximation in the CTF environment

        Args:
            num_actions (int): number of actions each agent can take (for CTF, this is 5 (stay still, up, down, left, right))
            batch_size (int): Number of transitions to be sampled from the replay buffer.
        '''

        super(DQN, self).__init__()
        self.batch_size = batch_size

        # set number of channels for the CNN
        self.c1 = 6
        self.c2 = 16
        self.c3 = 32

        # this CNN architecture will maintain the size of the observation throughout the convolution
        self.conv1 = nn.Conv2d(6, self.c1, 3, padding = 1)
        self.conv2 = nn.Conv2d(self.c1, self.c2, 3, padding = 1)
        self.conv3 = nn.Conv2d(self.c2, self.c3, 3, padding = 1)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(self.c3*num_obsv_states, num_actions)

    def forward(self, state):
        '''
        Propagates the state through the neural network to get q-values for each action

        Args:
            state (torch tensor): array of integers representing the grid-world with shape (batch_size, num_channels, num_agents, map_x, map_y)

        Returns:
            q_values (torch tensor): Q-values for the actions corresponding to the input state
        '''

        # TODO make it work for bool array for one_hot_encoder_v2 (or convert bool to int / float)
        # TODO for easier implementation, I have taken each observation on it's own (for 4 agents, and batch size 100, we
        # have 400 'states' going through the network)
        #
        # However, I think having a 3D convolution with the agent observations "stacked" for each transition
        # would also work and could be a cool thing to check out (for 4 agents, and batch size 100, we
        # have 100 'stacked states' going through the network)

        #TODO mess around with different network architectures (RNN, DenseNet, etc.)
        out = self.conv1(state)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)

        #TODO to optimize and allow all observations to be passed through the network at once, split into 4 vectors here?
        out = out.view(out.size(0), -1)

        q_values = self.fc(out)

        return q_values.cpu()

###############################
## other useful classes
class ReplayBuffer(object):
    def __init__(self, capacity):
        '''
        Inits the buffer as a deque

        Args:
            capacity (int): maximum capacity of the deque before entries are removed from the rear
        '''

        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        '''
        Appends a resized and formatted version of the transition tuple to the front of the replay buffer

        Args:
            state (np array): array of integers representing the current state of the grid-world with shape (num_agents, map_x, map_y, num_channels)
            action (list): list of actions for all agents at a timestep with shape (num_agents,)
            reward (int): reward for a single transition
            next_state (np array): array of integers representing the next state of the grid-world after action has been taken .  Has shape (num_agents, map_x, map_y, num_channels)
            done (bool): 0 -> the sim did not end on this transition, 1 -> the sim ended on this transition
        '''

        # swap dimensions so we have (batch_size, num_agents, num_channels, map_x, map_y)
        state = np.swapaxes(np.swapaxes(state, 1, 3), 2, 3)
        next_state = np.swapaxes(np.swapaxes(next_state, 1, 3), 2, 3)

        state = np.expand_dims(np.asarray(state), 0)
        action = np.expand_dims(np.asarray(action), 0)
        next_state = np.expand_dims(np.asarray(next_state), 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        '''
        Randomly samples transitions from the buffer

        Args:
            batch_size (int): number of transitions to be sampled

        Returns:
            state (torch.FloatTensor): batch of sampled states with shape (batch_size, num_agents, num_channels, map_x, map_y)
            action (torch.LongTensor): batch of sampled actions with shape (batch_size, num_agents)
            reward (torch.FloatTensor): batch of sampled rewards
            next_state (torch.FloatTensor): formatted next_state with shape (batch_size, num_agents, num_channels, map_x, map_y)
            done (np array): formatted done
        '''

        # state, action, reward, next_state, done = np.asarray(random.sample(self.buffer, batch_size))
        sample = np.asarray(random.sample(self.buffer, batch_size))

        state = np.vstack(sample[:, 0]) # gives (batch_size, num_agents, num_channels, map_x, map_y)
        action = np.vstack(sample[:, 1])
        reward = np.array(sample[:, 2], dtype = 'float')
        next_state = np.vstack(sample[:, 3])
        done = np.array(sample[:, 4])

        #TODO for the optimized version
        # state = np.concatenate(np.vstack(sample[:, 0]))  # gives (batch_size*num_agents, num_channels, map_x, map_y)
        # action = np.expand_dims(np.concatenate(np.vstack(sample[:, 1])), axis= 1)
        # reward = np.array(sample[:, 2], dtype = 'float')
        # next_state = np.concatenate(np.vstack(sample[:, 3]))
        # done = np.array(sample[:, 4])

        state = torch.from_numpy(state).type(torch.cuda.FloatTensor)
        next_state = torch.from_numpy(next_state).type(torch.cuda.FloatTensor)
        action = torch.from_numpy(action).type(torch.LongTensor)
        reward = torch.from_numpy(reward).type(torch.FloatTensor)

        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
