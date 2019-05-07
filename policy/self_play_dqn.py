'''
RL policy generator

'''

import torch
import numpy as np
import copy
from utility.utility_env import one_hot_encoder
from utility.utility_DQN import DQN, format_state_for_action


class PolicyGen:
    """Policy generator class for CtF env.

    This class can be used as a template for policy generator.
    Designed to summon an AI logic for the team of units.

    Methods:
        gen_action: Required method to generate a list of actions.
    """

    def __init__(self, self_play_network, train_params, free_map, agent_list, env):
        """Constuctor for policy class.

        This class can be used as a template for policy generator.

        Args:
            free_map (np.array): 2d map of static environment.
            agent_list (list): list of all friendly units.
        """
        self.model = copy.deepcopy(self_play_network)
        self.env = env
        self.train_params = train_params

    def gen_action(self, agent_list, observation, free_map=None):
        """Action generation method.

        This is a required method that generates list of actions corresponding
        to the list of units.

        Args:
            agent_list (list): list of all friendly units.
            observation (np.array): 2d map of partially observable map.
            free_map (np.array): 2d map of static environment (optional).

        Returns:
            action_out (list): list of integers as actions selected for team.
        """
        action_list = []
        state = one_hot_encoder(self.env._env, self.env.get_team_red, vision_radius = self.train_params['vision_radius'])

        state = format_state_for_action(state)
        with torch.no_grad():
            for i in range(len(agent_list)):
                q_values = self.model.forward(state[:, i, :, :, :])
                _, action = torch.max(q_values, 1)
                action_list.append(int(action.data))

        return action_list
