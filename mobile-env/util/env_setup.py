"""Utility module for setting up different envs"""
import numpy as np
import structlog
from shapely.geometry import Point
from ray.rllib.agents.ppo import DEFAULT_CONFIG
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from deepcomp.util.constants import SUPPORTED_ENVS, SUPPORTED_AGENTS, SUPPORTED_SHARING
from deepcomp.env.single_ue.variants import BinaryMobileEnv, DatarateMobileEnv, NormDrMobileEnv, RelNormEnv, MaxNormEnv
from deepcomp.env.multi_ue.central import CentralDrEnv, CentralNormDrEnv, CentralRelNormEnv, CentralMaxNormEnv
from deepcomp.env.multi_ue.multi_agent import MultiAgentMobileEnv, SeqMultiAgentMobileEnv
from deepcomp.env.entities.user import User
from deepcomp.env.entities.station import Basestation
from deepcomp.env.entities.map import Map
from deepcomp.env.util.movement import UniformMovement, RandomWaypoint
from deepcomp.util.callbacks import CustomMetricCallbacks


log = structlog.get_logger()


def get_env_class(env_type):
    """Return the env class corresponding to the string type (from CLI)"""
    assert env_type in SUPPORTED_AGENTS, f"Environment type was {env_type} but has to be one of {SUPPORTED_AGENTS}."

    if env_type == 'single':
        # return DatarateMobileEnv
        # return NormDrMobileEnv
        return RelNormEnv
    if env_type == 'central':
        # return CentralDrEnv
        # return CentralNormDrEnv
        return CentralRelNormEnv
        # return CentralMaxNormEnv
    if env_type == 'multi':
        return MultiAgentMobileEnv


def get_sharing_for_bs(sharing, bs_idx):
    """Return the sharing model for the given BS"""
    # if it's not mixed, it's the same for all BS
    if sharing != 'mixed':
        assert sharing in SUPPORTED_SHARING
        return sharing

    # else loop through the available sharing models
    sharing_list = ['resource-fair', 'rate-fair', 'proportional-fair']
    return sharing_list[bs_idx % len(sharing_list)]


def create_small_map(sharing_model):
    """
    Create small map and 2 BS

    :returns: tuple (map, bs_list)
    """
    map = Map(width=150, height=100)
    bs1 = Basestation('A', Point(50, 50), get_sharing_for_bs(sharing_model, 0))
    bs2 = Basestation('B', Point(100, 50), get_sharing_for_bs(sharing_model, 1))
    bs_list = [bs1, bs2]
    return map, bs_list


def create_dyn_small_map(sharing_model, bs_dist=100, dist_to_border=10):
    """Small env with 2 BS and dynamic distance in between"""
    map = Map(width=2 * dist_to_border + bs_dist, height=2 * dist_to_border)
    bs1 = Basestation('A', Point(dist_to_border, dist_to_border), sharing_model)
    bs2 = Basestation('B', Point(dist_to_border + bs_dist, dist_to_border), sharing_model)
    return map, [bs1, bs2]


def create_medium_map(sharing_model):
    """
    Deprecated: Use dynamic medium env instead. Kept this to reproduce earlier results.
    Same as large env, but with map restricted to areas with coverage.
    Thus, optimal episode reward should be close to num_ues * eps_length * 10 (ie, all UEs are always connected)
    """
    map = Map(width=205, height=85)
    bs1 = Basestation('A', Point(45, 35), sharing_model)
    bs2 = Basestation('B', Point(160, 35), sharing_model)
    bs3 = Basestation('C', Point(100, 85), sharing_model)
    bs_list = [bs1, bs2, bs3]
    return map, bs_list


def create_dyn_medium_map(sharing_model, bs_dist=100, dist_to_border=10):
    """
    Create map with 3 BS at equal distance. Distance can be varied dynamically. Map is sized automatically.
    Keep the same layout as old medium env here: A, B on same horizontal axis. C above in the middle
    """
    # calculate vertical distance from A, B to C using Pythagoras
    y_dist = np.sqrt(bs_dist ** 2 - (bs_dist / 2) ** 2)
    # derive map size from BS distance and distance to border
    map_width = 2 * dist_to_border + bs_dist
    map_height = 2 * dist_to_border + y_dist

    map = Map(width=map_width, height=map_height)
    # BS A is located at bottom left corner with specified distance to border
    bs1 = Basestation('A', Point(dist_to_border, dist_to_border), get_sharing_for_bs(sharing_model, 0))
    # other BS positions are derived accordingly
    bs2 = Basestation('B', Point(dist_to_border + bs_dist, dist_to_border), get_sharing_for_bs(sharing_model, 1))
    bs3 = Basestation('C', Point(dist_to_border + (bs_dist / 2), dist_to_border + y_dist), get_sharing_for_bs(sharing_model, 2))
    return map, [bs1, bs2, bs3]


def create_large_map(sharing_model):
    """
    Create larger map with 7 BS that are arranged in a typical hexagonal structure.

    :returns: Tuple(map, bs_list)
    """
    map = Map(width=230, height=260)
    bs_list = [
        # center
        Basestation('A', Point(115, 130), get_sharing_for_bs(sharing_model, 0)),
        # top left, counter-clockwise
        Basestation('B', Point(30, 80), get_sharing_for_bs(sharing_model, 1)),
        Basestation('C', Point(115, 30), get_sharing_for_bs(sharing_model, 2)),
        Basestation('D', Point(200, 80), get_sharing_for_bs(sharing_model, 3)),
        Basestation('E', Point(200, 180), get_sharing_for_bs(sharing_model, 4)),
        Basestation('F', Point(115, 230), get_sharing_for_bs(sharing_model, 5)),
        Basestation('G', Point(30, 180), get_sharing_for_bs(sharing_model, 6)),
    ]

    return map, bs_list


def create_dyn_large_map(sharing_model, num_bs, dist_to_border=10):
    assert 1 <= num_bs <= 7, "Only support 1-7 BS in large env"
    _, bs_list = create_large_map(sharing_model)
    # take only selected BS
    bs_list = bs_list[:num_bs]
    # create map with size according to BS positions
    max_x, max_y = None, None
    for bs in bs_list:
        if max_x is None or bs.pos.x > max_x:
            max_x = bs.pos.x
        if max_y is None or bs.pos.y > max_y:
            max_y = bs.pos.y
    map = Map(width=max_x + dist_to_border, height=max_y + dist_to_border)
    return map, bs_list


def create_ues(map, num_static_ues, num_slow_ues, num_fast_ues):
    """Create custom number of slow/fast UEs on the given map. Return UE list"""
    ue_list = []
    id = 1
    for i in range(num_static_ues):
        ue_list.append(User(str(id), map, pos_x='random', pos_y='random', movement=RandomWaypoint(map, velocity=0)))
        id += 1
    for i in range(num_slow_ues):
        ue_list.append(User(str(id), map, pos_x='random', pos_y='random', movement=RandomWaypoint(map, velocity='slow')))
        id += 1
    for i in range(num_fast_ues):
        ue_list.append(User(str(id), map, pos_x='random', pos_y='random', movement=RandomWaypoint(map, velocity='fast')))
        id += 1
    return ue_list


def create_custom_env(sharing_model):
    """Hand-created custom env. For demos or specific experiments."""
    # map with 4 BS at distance of 100; distance 10 to border of map
    map = Map(width=194, height=120)
    bs_list = [
        # left
        Basestation('A', Point(10, 60), get_sharing_for_bs(sharing_model, 0)),
        # counter-clockwise
        Basestation('B', Point(97, 10), get_sharing_for_bs(sharing_model, 1)),
        Basestation('C', Point(184, 60), get_sharing_for_bs(sharing_model, 2)),
        Basestation('D', Point(97, 110), get_sharing_for_bs(sharing_model, 3)),
    ]

    # UEs are now created dynamically according to CLI
    # ue_list = [
    #     User(str(1), map, pos_x=70, pos_y=40, movement=UniformMovement(map)),
    #     User(str(2), map, pos_x=80, pos_y=60, movement=UniformMovement(map))
    # ]
    return map, bs_list


def get_env(map_size, bs_dist, num_static_ues, num_slow_ues, num_fast_ues, sharing_model, num_bs=None):
    """Create and return the environment corresponding to the given map_size"""
    assert map_size in SUPPORTED_ENVS, f"Environment {map_size} is not one of {SUPPORTED_ENVS}."

    # create map and BS list
    map, bs_list = None, None
    if map_size == 'small':
        map, bs_list = create_small_map(sharing_model)
    elif map_size == 'medium':
        map, bs_list = create_dyn_medium_map(sharing_model, bs_dist=bs_dist)
    elif map_size == 'large':
        if num_bs is None:
            map, bs_list = create_large_map(sharing_model)
        else:
            map, bs_list = create_dyn_large_map(sharing_model, num_bs)
    elif map_size == 'custom':
        map, bs_list = create_custom_env(sharing_model)

    # create UEs
    ue_list = create_ues(map, num_static_ues, num_slow_ues, num_fast_ues)

    return map, ue_list, bs_list


def create_env_config(cli_args):
    """
    Create environment and RLlib config based on passed CLI args. Return config.

    :param cli_args: Parsed CLI args
    :return: The complete config for an RLlib agent, including the env & env_config
    """
    env_class = get_env_class(cli_args.agent)
    map, ue_list, bs_list = get_env(cli_args.env, cli_args.bs_dist, cli_args.static_ues, cli_args.slow_ues,
                                    cli_args.fast_ues, cli_args.sharing, cli_args.num_bs)

    # this is for DrEnv and step utility
    # env_config = {
    #     'episode_length': eps_length, 'seed': seed,
    #     'map': map, 'bs_list': bs_list, 'ue_list': ue_list, 'dr_cutoff': 'auto', 'sub_req_dr': True,
    #     'curr_dr_obs': False, 'ues_at_bs_obs': False, 'dist_obs': False, 'next_dist_obs': False
    # }
    # this is for the custom NormEnv and log utility
    env_config = {
        'episode_length': cli_args.eps_length, 'seed': cli_args.seed, 'map': map, 'bs_list': bs_list, 'ue_list': ue_list,
        'rand_episodes': cli_args.rand_train, 'new_ue_interval': cli_args.new_ue_interval, 'reward': cli_args.reward,
        'max_ues': cli_args.max_ues,
        # if enabled log_metrics: log metrics even during training --> visible on tensorboard
        # if disabled: log just during testing --> probably slightly faster training with less memory
        'log_metrics': True,
        # custom animation rendering
        'dashboard': cli_args.dashboard, 'ue_details': cli_args.ue_details,
    }

    # create and return the config
    config = DEFAULT_CONFIG.copy()
    # discount factor (default 0.99)
    # config['gamma'] = 0.5
    # 0 = no workers/actors at all --> low overhead for short debugging; 2+ workers to accelerate long training
    config['num_workers'] = cli_args.workers
    config['seed'] = cli_args.seed
    # write training stats to file under ~/ray_results (default: False)
    config['monitor'] = True
    config['train_batch_size'] = cli_args.batch_size        # default: 4000; default in stable_baselines: 128
    # auto normalize obserations by subtracting mean and dividing by std (default: "NoFilter")
    # config['observation_filter'] = "MeanStdFilter"
    # NN settings: https://docs.ray.io/en/latest/rllib-models.html#built-in-model-parameters
    # configure the size of the neural network's hidden layers; default: [256, 256]
    # config['model']['fcnet_hiddens'] = [512, 512, 512]
    # LSTM settings
    config['model']['use_lstm'] = cli_args.lstm
    # config['model']['lstm_use_prev_action_reward'] = True
    # config['log_level'] = 'INFO'    # ray logging default: warning
    # reset the env whenever the horizon/eps_length is reached
    config['horizon'] = cli_args.eps_length
    config['env'] = env_class
    config['env_config'] = env_config
    # callback for monitoring custom metrics
    config['callbacks'] = CustomMetricCallbacks
    config['log_level'] = 'ERROR'

    # for multi-agent env: https://docs.ray.io/en/latest/rllib-env.html#multi-agent-and-hierarchical
    if MultiAgentEnv in env_class.__mro__:
        # instantiate env to access obs and action space
        env = env_class(env_config)

        # use separate policies (and NNs) for each agent
        if cli_args.separate_agent_nns:
            # create policies also for all future UEs
            if env.max_ues > env.num_ue:
                log.warning("Varying num. UEs. Creating policy for all (future) UEs.",
                            curr_num_ue=env.num_ue, max_ues=env.max_ues)
                ue_ids = [str(i + 1) for i in range(env.max_ues)]
            else:
                ue_ids = [ue.id for ue in ue_list]

            config['multiagent'] = {
                # attention: ue.id needs to be a string! just casting it to str() here doesn't work;
                # needs to be consistent with obs keys --> easier, just use string IDs
                'policies': {ue_id: (None, env.observation_space, env.action_space, {}) for ue_id in ue_ids},
                'policy_mapping_fn': lambda agent_id: agent_id
            }
        # or: all UEs use the same policy and NN
        else:
            config['multiagent'] = {
                'policies': {'ue': (None, env.observation_space, env.action_space, {})},
                'policy_mapping_fn': lambda agent_id: 'ue'
            }

    return config

