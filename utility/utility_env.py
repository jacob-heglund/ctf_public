import numpy as np
import gym_cap.envs.const as CONST

UNKNOWN  = CONST.UNKNOWN          # -1
TEAM1_BG = CONST.TEAM1_BACKGROUND # 0
TEAM2_BG = CONST.TEAM2_BACKGROUND # 1
TEAM1_AG = CONST.TEAM1_UGV        # 2
TEAM1_UAV= CONST.TEAM1_UAV        # 3
TEAM2_AG = CONST.TEAM2_UGV        # 4
TEAM2_UAV= CONST.TEAM2_UAV        # 5
TEAM1_FL = CONST.TEAM1_FLAG       # 6
TEAM2_FL = CONST.TEAM2_FLAG       # 7
OBSTACLE = CONST.OBSTACLE         # 8
DEAD     = CONST.DEAD             # 9
SELECTED = CONST.SELECTED         # 10
COMPLETED= CONST.COMPLETED        # 11

def one_hot_encoder(state, agents, vision_radius=9, reverse=False):
    """Encoding pipeline for CtF state to one-hot representation

    6-channel one-hot representation of state.
    State is not binary: team2 is represented with -1.
    Channels are not symmetrical.

    :param state: CtF state in raw format
    :param agents: Agent list of CtF environment
    :param vision_radius: Size of the vision range (default=9)
    :param reverse:Reverse the color. Used for red-perspective (default=False)

    :return oh_state: One-hot encoded state
    """

    vision_lx = 2*vision_radius+1
    vision_ly = 2*vision_radius+1
    oh_state = np.zeros((len(agents),vision_lx,vision_ly,6))

    # team 1 : (1), team 2 : (-1), map elements: (0)
    map_channel = {UNKNOWN:0, DEAD:0,
                   TEAM1_BG:1, TEAM2_BG:1,
                   TEAM1_AG:2, TEAM2_AG:2,
                   TEAM1_UAV:3, TEAM2_UAV:3,
                   TEAM1_FL:4, TEAM2_FL:4,
                   OBSTACLE:5}
    if not reverse:
        map_color = {UNKNOWN:1, DEAD:0,
                     TEAM1_BG:0, TEAM2_BG:1,
                     TEAM1_AG:1, TEAM2_AG:-1,
                     TEAM1_UAV:1, TEAM2_UAV:-1,
                     TEAM1_FL:1, TEAM2_FL:-1,
                     OBSTACLE:1}
    else: # reverse color
        map_color = {UNKNOWN:1, DEAD:0,
                     TEAM1_BG:1, TEAM2_BG:0,
                     TEAM1_AG:-1, TEAM2_AG:1,
                     TEAM1_UAV:-1, TEAM2_UAV:1,
                     TEAM1_FL:-1, TEAM2_FL:1,
                     OBSTACLE:1}


    # Expand the observation with wall to avoid dealing with the boundary
    sx, sy = state.shape
    _state = np.full((sx+2*vision_radius, sy+2*vision_radius),OBSTACLE)
    _state[vision_radius:vision_radius+sx, vision_radius:vision_radius+sy] = state
    state = _state

    for idx,agent in enumerate(agents):
        # Initialize Variables
        x, y = agent.get_loc()
        x += vision_radius
        y += vision_radius
        vision = state[x-vision_radius:x+vision_radius+1,y-vision_radius:y+vision_radius+1] # extract view

        # FULL MATRIX OPERATION
        for channel, val in map_color.items():
            if val == 1:
                oh_state[idx,:,:,map_channel[channel]] += (vision == channel).astype(np.int32)
            elif val == -1:
                oh_state[idx,:,:,map_channel[channel]] -= (vision == channel).astype(np.int32)

    return oh_state

def one_hot_encoder_v2(state, agents, vision_radius=9, reverse=False):
    """ Encoding pipeline for CtF state to one-hot representation
    11-channel one-hot representation of state.
    State is binary.
    Some optimization is included.

    :param state: CtF state in raw format
    :param agents: Agent list of CtF environment
    :param vision_radius: Size of the vision range (default=9)`
    :param reverse:Reverse the color. Used for red-perspective (default=False)

    :return oh_state: One-hot encoded state
    """

    num_channel = 11
    num_agents = len(agents)

    vision_lx = 2*vision_radius+1
    vision_ly = 2*vision_radius+1

    # Map channel for each elements
    if not reverse:
        order = [UNKNOWN, OBSTACLE, TEAM1_BG, TEAM2_BG, TEAM1_AG, TEAM2_AG,
                                  TEAM1_UAV, TEAM2_UAV, TEAM1_FL, TEAM2_FL, DEAD]
    else:
        order = [UNKNOWN, OBSTACLE, TEAM2_BG, TEAM1_BG, TEAM2_AG, TEAM1_AG,
                                  TEAM2_UAV, TEAM1_UAV, TEAM2_FL, TEAM1_FL, DEAD]
    map_channel = dict(zip(order, range(num_channel)))

    # Padding Boundary
    #state = np.pad(state, ((vision_radius,vision_radius),(vision_radius,vision_radius)), 'constant', constant_values=OBSTACLE)
    sx, sy = state.shape
    _state = np.full((sx+2*vision_radius, sy+2*vision_radius),OBSTACLE)
    _state[vision_radius:vision_radius+sx, vision_radius:vision_radius+sy] = state
    state = _state

    each_agent = []
    for idx, agent in enumerate(agents):
        # Initialize Variables
        x, y = agent.get_loc()
        x += vision_radius
        y += vision_radius
        vision = state[x-vision_radius:x+vision_radius+1,y-vision_radius:y+vision_radius+1] # extract view

        # operation
        each_channel = []
        for element, channel in map_channel.items():
            each_channel.append(vision==element)
        each_agent.append(np.stack(each_channel, axis=-1))
    oh_state = np.stack(each_agent, axis=0)

    return oh_state


# Debug
def debug():
    """debug
    Include testing code for above methods and classes.
    The execution will start witn __main__, and call this method.
    """

    import gym
    import time
    env = gym.make("cap-v0")
    s = env.reset(map_size=20)

    print('start running')
    stime = time.time()
    for _ in range(3000):
        s = env.reset(map_size=20)
        one_hot_encoder(s, env.get_team_blue)
    print(f'Finish testing for one-hot-encoder: {time.time()-stime} sec')

    s = env.reset(map_size=20)

    print('start running v2')
    stime = time.time()
    for _ in range(3000):
        s = env.reset(map_size=20)
        one_hot_encoder_v2(s, env.get_team_blue)
    print(f'Finish testing for one-hot-encoder: {time.time()-stime} sec')

if __name__ == '__main__':
    debug()
