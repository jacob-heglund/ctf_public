def hindsight_encoder(trajectory):
    """
    Description:
    takes a team's trajectory and converts the full map at each timestep to a modified map
    where the goal (s') is always achieved by taking action a from state s

    Args:
        trajectory (list): list of transition tuples from the "real" episode (modified_state, action, reward, modified_next_state, done)
            - modified_state (tuple): (full_map, one_hot_state, global_position)
            - modified_next_state (tuple): (full_map, one_hot_state, global_position)
                - full_map (np.array): env._env, the full map before onehot is applied
                - one_hot_state (np.array): one_hot_encoder(env._env), state separated into channels for different features
                - global_position (np.array): call agent.get_loc(), 2d coordinates on the map
    """
    # TODO implement for multiple agents too (collect the trajectories of each agent and do stuff here)

    num_steps = len(trajectory)
    # get the agent's global state at each time-step of the trajectory
    #TODO set conditions so the enemy flag can't be placed in friendly territory
    for i in range(num_steps-1):
        map_curr = trajectory[i][0][0]

        # change the map for current state
        # replace the old goal by enemy red-territory background
        #TODO this won't work in general when enemies are introduced since they can cover the flag
        goal_global_position = np.where(map_curr == TEAM2_FL)
        map_curr[goal_global_position] = TEAM2_BG
        updated_map_curr = old_map_curr

        # modify new_map with the enemy flag moved to the agent's next_state (global_position at time t+1)
        new_goal_global_position = trajectory[i+1][0][2]
        updated_map_curr[new_goal_global_position] = TEAM2_FL

        # change the map for next_state
        # remove the flag from it's actual position
        # don't  place the flag since our agent would be covering it on the map
        updated_map_next = trajectory[i+1][0][0]
        updated_map_next[goal_global_position] = TEAM2_BG

        # reward the agent appropriately for reaching the flag
        reward = 1.0

        # set state and next_state as the one-hot encoded versions of the maps
        state = one_hot_encoder(updated_map_curr, env.get_team_blue, vision_radius = train_params['vision_radius'])
        next_state = one_hot_encoder(updated_map_next, env.get_team_blue, vision_radius = train_params['vision_radius'])

        # push new trajectories into the replay buffer
        replay_buffer.push(state, action, reward, next_state, done)
