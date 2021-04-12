import sys
import os
import time
import logging
from datetime import datetime
from collections import defaultdict
from pathlib import Path

import pandas as pd
import structlog
import matplotlib.pyplot as plt
import matplotlib.animation
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import ray
import ray.tune
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from deepcomp.util.constants import SUPPORTED_ALGS, SUPPORTED_RENDER, get_result_dirs
from deepcomp.agent.dummy import RandomAgent, FixedAgent
from deepcomp.agent.heuristics import Heuristic3GPP, FullCoMP, DynamicSelection
from deepcomp.agent.brute_force import BruteForceAgent
from deepcomp.util.logs import config_logging


class Simulation:
    """Simulation class for training and testing agents."""
    def __init__(self, config, agent_name, cli_args, debug=False):
        """
        Create a new simulation object to hold the agent and environment, train & test & visualize the agent + env.

        :param config: RLlib agent config
        :param agent_name: String identifying the agent
        :param cli_args: Dict of CLI args
        :param debug: Whether or not to enable ray's local_mode for debugging
        """
        # config and env
        self.config = config
        self.env_class = config['env']
        self.env_name = config['env'].__name__
        self.env_config = config['env_config']
        self.episode_length = self.env_config['episode_length']
        # detect automatically if the env is a multi-agent env by checking all (not just immediate) ancestors
        self.multi_agent_env = MultiAgentEnv in self.env_class.__mro__
        # num workers for parallel execution of eval episodes
        self.num_workers = config['num_workers']
        self.cli_args = cli_args
        self.dashboard = self.env_config['dashboard']

        # agent
        assert agent_name in SUPPORTED_ALGS, f"Agent {agent_name} not supported. Supported agents: {SUPPORTED_ALGS}"
        self.agent_name = agent_name
        self.agent = None
        self.agent_train_steps = None
        # only init ray if necessary --> lower overhead for dummy agents
        if self.agent_name == 'ppo':
            if self.cli_args.cluster:
                # when running on a cluster (with "ray up"), auto connect to Redis and cluster head
                ray.init(address='auto')
            else:
                # disable for local execution (it won't find a cluster and will break). optionally, enable local debug
                # also less logging and no dashboard
                ray.init(local_mode=debug, include_dashboard=False, log_to_driver=False)
        self.agent_path = None

        # filename for saving is set when loading the agent
        self.result_filename = None
        # result dirs for saving progress, test results, and videos
        self.result_dir, self.train_dir, self.test_dir, self.video_dir = get_result_dirs(result_dir=cli_args.result_dir)

        self.log = structlog.get_logger()
        self.log.debug('Simulation init', env=self.env_name, eps_length=self.episode_length, agent=self.agent_name,
                       multi_agent=self.multi_agent_env, num_workers=self.num_workers)

    @staticmethod
    def extract_agent_id(agent_path):
        """Extract and return agent ID from path. Eg, 'PPO_MultiAgentMobileEnv_14c68_00000_0_2020-10-22_10-03-33'"""
        if agent_path is not None:
            # walk through parts of path and return the one starting with 'PPO_'
            parts = os.path.normpath(agent_path).split(os.sep)
            for p in parts:
                if p.startswith('PPO_'):
                    return p
        return None

    @property
    def metadata(self):
        """Dict with metadata about the simulation"""
        # distinguish multi-agent RL with separate NNs rather than a shared NN for all agents
        agent_str = self.cli_args.agent
        if agent_str == 'multi' and self.cli_args.separate_agent_nns:
            agent_str = 'multi-sep-nns'

        data = {
            'alg': self.cli_args.alg,
            'agent': agent_str,
            'agent_path': self.agent_path,
            'agent_id': self.extract_agent_id(self.agent_path),
            'env': self.env_name,
            'env_size': self.cli_args.env,
            'eps_length': self.episode_length,
            'num_bs': len(self.env_config['bs_list']),
            'sharing_model': self.cli_args.sharing,
            'num_ue_static': self.cli_args.static_ues,
            'num_ue_slow': self.cli_args.slow_ues,
            'num_ue_fast': self.cli_args.fast_ues,
            'result_filename': self.result_filename,
        }

        # add training iteration
        if data['alg'] == 'ppo':
            data['train_iteration'] = self.agent.iteration
            # not sure how to access the actual training steps or whether that's even possible

        return data

    def train(self, stop_criteria, restore_path=None, scheduler=None):
        """
        Train an RLlib agent using tune until any of the configured stopping criteria is met.

        :param stop_criteria: Dict with stopping criteria.
            See https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run
        :param restore_path: Path to trained agent to continue training (if any)
            The agent's latest checkpoint is loaded automatically
            The trained agent needs to have the same settings and scenario for continuing training
            When continuing training, the number of training steps continues too, ie, is not reset to 0 after restoring
        :return: Return the path to the saved agent (checkpoint) and tune's ExperimentAnalysis object
            See https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis-tune-experimentanalysis
        """
        # load latest checkpoint within the given agent's directory
        if restore_path is not None:
            restore_path = self.get_last_checkpoint_path(restore_path)

        analysis = ray.tune.run(PPOTrainer, config=self.config, local_dir=self.train_dir, stop=stop_criteria,
                                # checkpoint every 10 iterations and at the end; keep the best 10 checkpoints
                                checkpoint_at_end=True, checkpoint_freq=10, keep_checkpoints_num=10,
                                checkpoint_score_attr='episode_reward_mean', restore=restore_path,
                                scheduler=scheduler)
        analysis.default_metric = 'episode_reward_mean'
        analysis.default_mode = 'max'

        # tune returns an ExperimentAnalysis that can be cast to a Pandas data frame
        # object https://docs.ray.io/en/latest/tune/api_docs/analysis.html#experimentanalysis
        df = analysis.dataframe()
        checkpoint_path = analysis.get_best_checkpoint(trial=analysis.get_best_trial())
        self.log.info('Training done', timesteps_total=int(df['timesteps_total']),
                      episodes_total=int(df['episodes_total']), episode_reward_mean=float(df['episode_reward_mean']),
                      num_steps_sampled=int(df['info/num_steps_sampled']),
                      num_steps_trained=int(df['info/num_steps_trained']),
                      log_dir=analysis.get_best_logdir())

        # plot results
        # this only contains (and plots) the last 100 episodes --> not useful
        #  --> use tensorboard instead; or read and plot progress.csv
        # eps_results = df['hist_stats']
        # self.plot_learning_curve(eps_results['episode_lengths'], eps_results['episode_reward'])
        return checkpoint_path, analysis

    @staticmethod
    def get_specific_checkpoint(rllib_dir):
        """
        Return path to checkpoint file if rllib_dir points to a specific checkpoint folder (or file).
        Else return None.
        """
        if 'checkpoint' not in rllib_dir:
            return None
        # if it directly points to the checkpoint file, just return it
        if os.path.isfile(rllib_dir):
            return rllib_dir
        # if it only points to the checkpoint folder, derive the checkpoint file and return it
        checkpoint_number = rllib_dir.split('_')[-1]
        return os.path.join(rllib_dir, f'checkpoint-{checkpoint_number}')

    @staticmethod
    def get_last_checkpoint_path(rllib_dir):
        """Given an RLlib training dir, return the full path to the last checkpoint"""
        # check if rllib_dir is really already a pointer to a specific checkpoint; in that case, just return it
        if 'checkpoint' in rllib_dir:
            return Simulation.get_specific_checkpoint(rllib_dir)

        rllib_dir = os.path.abspath(rllib_dir)
        checkpoints = [f for f in os.listdir(rllib_dir) if f.startswith('checkpoint')]
        # sort according to checkpoint number after '_'
        sorted_checkpoints = sorted(checkpoints, key=lambda cp: int(cp.split('_')[-1]))
        last_checkpoint_dir = os.path.join(rllib_dir, sorted_checkpoints[-1])
        # eg, retrieve '10' from '...PPO_MultiAgentMobileEnv_0_2020-07-14_17-28-33je5r1lov/checkpoint_10'
        last_checkpoint_no = last_checkpoint_dir.split('_')[-1]
        # construct full checkpoint path, eg, '...r1lov/checkpoint_10/checkpoint-10'
        last_checkpoint_path = os.path.join(last_checkpoint_dir, f'checkpoint-{last_checkpoint_no}')
        return last_checkpoint_path

    @staticmethod
    def get_best_checkpoint_path(rllib_dir):
        """Given an RLlib training dir, return the full path of the best checkpoint"""
        # check if rllib_dir is really already a pointer to a specific checkpoint; in that case, just return it
        if 'checkpoint' in rllib_dir:
            return Simulation.get_specific_checkpoint(rllib_dir)

        rllib_dir = os.path.abspath(rllib_dir)
        analysis = ray.tune.Analysis(rllib_dir)
        analysis.default_metric = 'episode_reward_mean'
        # analysis.default_metric = 'custom_metrics/sum_utility_mean'
        analysis.default_mode = 'max'
        checkpoint = analysis.get_best_checkpoint(analysis._get_trial_paths()[0])
        return os.path.abspath(checkpoint)

    def get_training_steps(self):
        """Get and return the total training steps of the currently loaded agent"""
        if self.agent_name != 'ppo':
            return None
        # usually train steps = train iterations * batch size, but it also somehow depends on num. workers
        # --> better to read value from progress.csv
        progress_dir = Path(self.agent_path).parent.parent
        progress_file = os.path.join(progress_dir, 'progress.csv')
        df = pd.read_csv(progress_file)
        train_steps = int(df[df['training_iteration'] == self.agent.training_iteration]['timesteps_total'])
        self.log.info("Loaded training progress", train_iteration = self.agent.training_iteration,
                      train_steps=train_steps, progress_file=progress_file)
        return train_steps

    def load_agent(self, rllib_dir=None, rand_seed=None, fixed_action=1, explore=False):
        """
        Load a trained RLlib agent from the specified rllib_path. Call this before testing a trained agent.

        :param rllib_dir: Path pointing to the agent's training dir (only used for RLlib agents)
        :param rand_seed: RNG seed used by the random agent (ignored by other agents)
        :param fixed_action: Fixed action performed by the fixed agent (ignored by the others)
        :param explore: Whether to keep exploration enabled. Set to False when testing an RLlib agent.
        True for continuing training.
        """
        checkpoint_path = None
        if self.agent_name == 'ppo':
            # turn off exploration for testing the loaded agent
            self.config['explore'] = explore
            self.agent = PPOTrainer(config=self.config, env=self.env_class)
            # TODO: changed to last checkpoint rather then best for demo --> back to best later
            # self.agent_path = self.get_best_checkpoint_path(rllib_dir)
            self.agent_path = self.get_last_checkpoint_path(rllib_dir)
            self.log.info('Loading PPO agent', checkpoint=self.agent_path)
            try:
                self.agent.restore(self.agent_path)
            except (AssertionError, ValueError) as e:
                self.log.error(f"Error loading agent. Mismatch of neural network size and number of UEs or env size"
                               f" when using a pretrained central DeepCoMP agent? Error: '{str(e)}'")
                sys.exit()
        if self.agent_name == '3gpp':
            self.agent = Heuristic3GPP()
        if self.agent_name == 'fullcomp':
            self.agent = FullCoMP()
        if self.agent_name == 'dynamic':
            self.agent = DynamicSelection(epsilon=0.8)
        if self.agent_name == 'brute-force':
            self.agent = BruteForceAgent(self.num_workers)
        if self.agent_name == 'random':
            # instantiate the environment to get the action space
            env = self.env_class(self.env_config)
            self.agent = RandomAgent(env.action_space, seed=rand_seed)
        if self.agent_name == 'fixed':
            self.agent = FixedAgent(action=fixed_action, noop_interval=100)

        self.log.info('Agent loaded', agent=type(self.agent).__name__, rllib_dir=rllib_dir, checkpoint=checkpoint_path)

        # set a suitable filename for saving testing videos and results later
        self.set_result_filename()

        # read the number of training steps
        self.agent_train_steps = self.get_training_steps()

    def set_result_filename(self):
        """Return a suitable filename (without file ending) in the format 'agent_env-class_env-size_num-ues_time'"""
        assert self.agent is not None, "Set the filename after loading the agent"
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        agent_name = type(self.agent).__name__
        env_size = self.cli_args.env
        num_ues = self.cli_args.static_ues + self.cli_args.slow_ues + self.cli_args.fast_ues
        train = 'rand' if self.cli_args.rand_train else 'fixed'
        test = 'rand' if self.cli_args.rand_test else 'fixed'
        seed = self.cli_args.seed
        self.result_filename = \
            f'{agent_name}_{self.env_name}_{env_size}_{self.cli_args.sharing}_{num_ues}UEs-{self.cli_args.reward}' \
            f'_{train}-{test}_{seed}_{timestamp}'

    def save_animation(self, fig, patches, mode):
        """
        Create and save matplotlib animation

        :param fig: Matplotlib figure
        :param patches: List of patches to draw for each step in the animation
        :param mode: How to save the animation. Options: 'video' (=html5) or 'gif' (requires ImageMagick)
        """
        render_modes = SUPPORTED_RENDER - {None}
        assert mode in render_modes, f"Render mode {mode} not in {render_modes}"
        anim = matplotlib.animation.ArtistAnimation(fig, patches, interval=100, repeat=False)

        # save html5 video
        if mode == 'html' or mode == 'both':
            html = anim.to_html5_video()
            with open(f'{self.video_dir}/{self.result_filename}.html', 'w') as f:
                f.write(html)
            self.log.info('Video saved', path=f'{self.video_dir}/{self.result_filename}.html')

        # save gif; requires external dependency ImageMagick
        if mode == 'gif' or mode == 'both':
            try:
                anim.save(f'{self.video_dir}/{self.result_filename}.gif', writer='imagemagick')
                self.log.info('Gif saved', path=f'{self.video_dir}/{self.result_filename}.gif')
            except TypeError:
                self.log.error('ImageMagick needs to be installed for saving gifs.')

    def apply_action_single_agent(self, obs, env, state=None):
        """
        For the given observation and a trained/loaded agent, get and apply the next action. Only single-agent envs.

        :param dict obs: Dict of observations for all agents
        :param env: The environment to which to apply the actions to
        :param state: Optional state of the RNN/LSTM if used
        :returns: tuple (obs, r, done, info, state) WHERE
            obs is the next observation
            r is the immediate reward
            done is done['__all__'] indicating if all agents are done
        """
        assert not self.multi_agent_env, "Use apply_action_multi_agent for multi-agent envs"
        assert self.agent is not None, "Train or load an agent before running the simulation"
        # normal MLP NN
        if state is None:
            action = self.agent.compute_action(obs)
        # RNN/LSTM, which requires state
        else:
            action, state, logits = self.agent.compute_action(obs, state=state)
        next_obs, reward, done, info = env.step(action)
        self.log.debug("Step", t=info['time'], obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        return next_obs, reward, done, info, state

    def apply_action_multi_agent(self, obs, env, state=None):
        """
        Same as apply_action_single_agent, but for multi-agent envs. For each agent, unpack obs & choose action,
        before applying it to the env.

        :param dict obs: Dict of observations for all agents
        :param env: The environment to which to apply the actions to
        :param state: Optional state of the RNN/LSTM if used
        :returns: tuple (obs, r, done, info, state) WHERE
            obs is the next observation
            r is the summed up immediate reward for all agents
            done is done['__all__'] indicating if all agents are done
        """
        assert self.multi_agent_env, "Use apply_action_single_agent for single-agent envs"
        assert self.agent is not None, "Train or load an agent before running the simulation"
        action = {}
        for agent_id, agent_obs in obs.items():
            policy_id = self.config['multiagent']['policy_mapping_fn'](agent_id)
            # normal MLP NN
            if state is None:
                action[agent_id] = self.agent.compute_action(agent_obs, policy_id=policy_id)
            # RNN/LSTM, which requires state
            else:
                action[agent_id], state, logits = self.agent.compute_action(agent_obs, policy_id=policy_id, state=state)
        next_obs, reward, done, info = env.step(action)
        # info is currently the same for all agents; just get the first one
        info = list(info.values())[0]
        self.log.debug("Step", t=info['time'], obs=obs, action=action, reward=reward, next_obs=next_obs,
                       done=done['__all__'])
        return next_obs, sum(reward.values()), done['__all__'], info, state

    @property
    def agent_dashboard_name(self):
        """Map self.agent_name to some prettier name for the dashboard"""
        if self.agent_name == 'ppo':
            if self.cli_args.agent == 'central':
                return 'DeepCoMP'
            elif self.cli_args.agent == 'multi':
                if self.cli_args.separate_agent_nns:
                    return 'D3-CoMP'
                else:
                    return 'DD-CoMP'
            return None
        elif self.agent_name == '3gpp':
            return '3GPP'
        elif self.agent_name == 'fullcomp':
            return 'FullCoMP'
        elif self.agent_name == 'dynamic':
            return 'Dynamic CoMP'
        elif self.agent_name == 'brute-force':
            return 'Brute Force (Opt.)'
        elif self.agent_name == 'random':
            return 'Random'
        elif self.agent_name == 'fixed':
            return 'Fixed'
        self.log.warning("Unknown agent name", agent_name=self.agent_name)
        return self.agent_name

    def setup_dashboard(self, map, ue_ids):
        """
        Setup the dashboard with matplotlib figures and gridspec; based on the map size

        :param map: The environment map, used for dimensioning the plot and dashboard
        :param ue_ids: List of UE IDs for which to show stats on the dashboard
        :return: Matplotlib figure, main axis, and dict of other axes
        """
        if len(ue_ids) > 3:
            self.log.info("Showing many UEs on dashboard may look bad. Recommended: 2 UEs.", num_ues=ue_ids)

        fig = plt.figure(constrained_layout=True, figsize=map.dashboard_figsize)
        gs = fig.add_gridspec(4, 3)
        # main map view
        ax_main = fig.add_subplot(gs[:, :2])

        # text box with general info (top right)
        ax_text = fig.add_subplot(gs[0, 2])
        ax_text.axis('off')

        # global stats (right; below text box)
        ax_avg = fig.add_subplot(gs[1, 2])
        ax_avg.set_title('Avg. QoE')
        ax_avg.set_ylim(-20, 20)

        # UE-specific stats (right; below global)
        ue_axes = {}
        row = 2
        for ue_id in ue_ids:
            ue_axes[ue_id] = fig.add_subplot(gs[row, 2], sharex=ax_avg, sharey=ax_avg)
            ue_axes[ue_id].set_title(f'UE {ue_id}: QoE')
            row += 1
        # set xlabel for last axis
        ue_axes[ue_id].set_xlabel('Time')

        # prepare dict of dashboard_axes
        dashboard_axes = {
            'main': ax_main, 'text': ax_text, 'avg': ax_avg, 'ue': ue_axes
        }

        return fig, ax_main, dashboard_axes

    def get_dashboard_data(self, dashboard_axes, scalar_metrics, vector_metrics):
        """Return a structured dict with data for plotting on the dashboard"""
        # get number of training steps (if applicable)
        train_steps = 'N/A'
        if self.agent_train_steps is not None:
            train_steps = self.agent_train_steps

        num_ue = self.cli_args.static_ues + self.cli_args.slow_ues + self.cli_args.fast_ues
        dashboard_data = {
            'agent': self.agent_dashboard_name,
            'train_steps': train_steps,
            'avg': [t['sum_utility'] / num_ue for t in scalar_metrics],
            'ue': {}
        }
        # fill UE-specific data
        if 'ue' in dashboard_axes:
            for ue_id in dashboard_axes['ue'].keys():
                dashboard_data['ue'][ue_id] = [t['utility'][f'UE {ue_id}'] for t in vector_metrics]
        return dashboard_data

    def run_episode(self, env, render=None, log_dict=None):
        """
        Run a single episode on the given environment. Append episode reward and exec time to list and return.

        :param env: Instance of the environment to use (each joblib iteration will still use its own instance)
        :param render: Whether/How to render the episode
        :param log_dict: Dict with logging levels to set
        :return: Tuple of eps_duration (scalar), step rewards (list), metrics per step (list of dicts)
        """
        # list of rewards and metrics (which are a dict) for each time step
        rewards = []
        scalar_metrics = []
        vector_metrics = []

        # no need to instantiate new env since each joblib iteration has its own copy
        # that's why we need to set the logging level again for each iteration
        config_logging()
        if log_dict is not None:
            env.set_log_level(log_dict)

        eps_start = time.time()
        if render is not None:
            dashboard_axes = None
            # dashboard_data = None
            dashboard_data = self.get_dashboard_data({}, scalar_metrics, vector_metrics)
            if self.dashboard:
                # show extra stats of first two UEs on dashboard
                ue_ids = [ue.id for ue in env.ue_list][:2]
                fig, ax_main, dashboard_axes = self.setup_dashboard(env.map, ue_ids)
                dashboard_data = self.get_dashboard_data(dashboard_axes, scalar_metrics, vector_metrics)
            else:
                fig = plt.figure(figsize=env.map.figsize)
                ax_main = plt.gca()
            # equal aspect ratio to avoid distortions
            ax_main.set_aspect('equal')

        # run until episode ends
        patches = []
        t = 0
        done = False
        obs = env.reset()
        # if using brute-force agent, pass the environment
        if self.agent_name == 'brute-force':
            self.agent.env = env
        # init state for LSTM: https://github.com/ray-project/ray/issues/9220#issuecomment-652146377
        state = None
        if self.config['model']['use_lstm']:
            cell_size = self.config['model']['lstm_cell_size']
            state = [np.zeros(cell_size), np.zeros(cell_size)]
        # for continuous problems, stop evaluation after fixed eps length
        while (done is None or not done) and t < self.episode_length:
            if render is not None:
                # update extra data for dashboard
                if self.dashboard:
                    dashboard_data = self.get_dashboard_data(dashboard_axes, scalar_metrics, vector_metrics)
                patches.append(env.render(ax=ax_main, dashboard_axes=dashboard_axes, dashboard_data=dashboard_data))
                if render == 'plot':
                    plt.show()

            # get and apply action
            if self.multi_agent_env:
                obs, reward, done, info, state = self.apply_action_multi_agent(obs, env, state)
            else:
                obs, reward, done, info, state = self.apply_action_single_agent(obs, env, state)
            t = info['time']

            # save reward and metrics
            rewards.append(reward)
            scalar_metrics.append(info['scalar_metrics'])
            vector_metrics.append(info['vector_metrics'])

        # create the animation
        if render is not None:
            if not self.dashboard:
                fig.tight_layout()
            self.save_animation(fig, patches, render)

        # episode time in seconds (to measure simulation efficiency)
        eps_duration = time.time() - eps_start

        self.log.debug('Episode complete', eps_duration=eps_duration, avg_step_reward=np.mean(rewards),
                       scalar_metrics=list(scalar_metrics[0].keys()), vector_metrics=list(vector_metrics[0].keys()))
        return eps_duration, rewards, scalar_metrics, vector_metrics

    @staticmethod
    def summarize_scalar_results(eps_duration, rewards, scalar_metrics):
        """
        Summarize given results into single result dict containing everything that should be logged and written to file.

        :param eps_duration: List of episode durations (in s)
        :param rewards: List of lists with rewards per step per episode
        :param scalar_metrics: List of lists, containing a dict of metric --> value for each episode for each time step
        :returns: Dict of result name --> whatever should be logged and saved (eg, mean, std, etc)
        """
        results = defaultdict(list)
        num_episodes = len(eps_duration)
        # get metric names from first metric dict (first episode, first step); it's the same for all steps and eps
        metric_names = list(scalar_metrics[0][0].keys())

        # iterate over all episodes and aggregate the results per episode
        for e in range(num_episodes):
            # add episode, eps_duration and rewards
            results['episode'].append(e)
            results['eps_duration_mean'].append(eps_duration[e])
            results['eps_duration_std'].append(eps_duration[e])
            results['step_reward_mean'].append(np.mean(rewards[e]))
            results['step_reward_std'].append(np.std(rewards[e]))

            # calc mean and std per metric and episode
            for metric in metric_names:
                metric_values = [scalar_metrics[e][t][metric] for t in range(len(scalar_metrics[e]))]
                results[f'{metric}_mean'].append(np.mean(metric_values))
                results[f'{metric}_std'].append(np.std(metric_values))

        # convert defaultdict to normal dict
        return dict(results)

    def write_scalar_results(self, scalar_results):
        """Write experiment results to CSV file. Include all relevant info."""
        result_file = f'{self.test_dir}/{self.result_filename}.csv'
        self.log.info("Writing scalar results", file=result_file)

        data = self.metadata
        # training data for PPO
        if self.agent_name == 'ppo':
            data.update({
                'train_steps': self.cli_args.train_steps,
                'train-iter': self.cli_args.train_iter,
                'target_reward': self.cli_args.target_reward,
                'target-utility': self.cli_args.target_utility,
            })

        # add actual results and save to file
        data.update(scalar_results)
        df = pd.DataFrame(data=data)
        df.to_csv(result_file)

    def write_vector_results(self, vector_metrics):
        """
        Write vector metrics into a data frames and save them to pickle, incl. meta data/attributes.
        One data frame and pickle file per metric.
        Vector metrics contain measurements per UE per time step (per evaluation episode).

        :param vector_metrics: List of lists of dicts of dicts: One list per episode with dicts per time step.
        Each dict maps metric name to another dict, which again maps UE ID to the metric value.
        :return: list of result dicts
        """
        # in case there are not vector metrics
        if len(vector_metrics) == 0 or len(vector_metrics[0]) == 0:
            return []

        # construct separate dfs per metric
        dfs = []
        metrics = list(vector_metrics[0][0].keys())
        for metric in metrics:
            # init dict with empty lists
            data = {'episode': [], 'time_step': []}
            ues = list(vector_metrics[-1][-1][metric].keys())
            for ue in ues:
                data[ue] = []

            # fill dict with values from vector_metrics
            for eps, eps_dict in enumerate(vector_metrics):
                for step, step_dict in enumerate(eps_dict):
                    data['episode'].append(eps)
                    data['time_step'].append(step)
                    metric_dict = step_dict[metric]
                    for ue in ues:
                        if ue in metric_dict:
                            data[ue].append(metric_dict[ue])
                        else:
                            data[ue].append(None)

            # create and write data frame
            df = pd.DataFrame(data)
            df.attrs = self.metadata
            df.attrs['metric'] = metric
            df.attrs['num_episodes'] = len(vector_metrics)
            df.attrs['env_config'] = self.env_config
            df.attrs['cli_args'] = vars(self.cli_args)
            dfs.append(df)
            result_file = f'{self.test_dir}/{self.result_filename}_{metric}.pkl'
            self.log.info('Writing vector results', metric=metric, file=result_file)
            df.to_pickle(result_file)

        return dfs

    def run(self, num_episodes=1, render=None, log_dict=None, write_results=False):
        """
        Run one or more simulation episodes. Render situation at beginning of each time step. Return episode rewards.

        :param int num_episodes: Number of episodes to run
        :param str render: If and how to render the simulation. Options: None, 'plot', 'video', 'gif'
        :param dict log_dict: Dict of logger names --> logging level used to configure logging in the environment
        :param bool write_results: Whether or not to write experiment results to file
        :return list: Return list of lists with step rewards for all episodes
        """
        assert self.agent is not None, "Train or load an agent before running the simulation"
        assert (num_episodes == 1) or (render is None), "Turn off rendering when running for multiple episodes"
        if self.num_workers > 1:
            # parallel evaluation doesn't work for PPO and brute force; the heuristics are fast anyways
            self.log.warning("Evaluating with a single worker for reproducibility and compatibility.")
            self.num_workers = 1
        assert self.num_workers == 1, "Evaluation needs to be done with a single worker"

        # enable metrics logging, configure episode randomization, instantiate env, and set logging level
        self.env_config['log_metrics'] = True
        self.env_config['rand_episodes'] = self.cli_args.rand_test
        env = self.env_class(self.env_config)
        if log_dict is not None:
            env.set_log_level(log_dict)

        # simulate episodes in parallel; show progress with tqdm if running for more than one episode
        self.log.info('Starting evaluation', num_episodes=num_episodes, num_workers=self.num_workers,
                      static_ues=self.cli_args.static_ues, slow_ues=self.cli_args.slow_ues,
                      fast_ues=self.cli_args.fast_ues)
        # there is currently no parallelization; eval is limited to a single worker
        # run episodes in parallel using joblib
        zipped_results = Parallel(n_jobs=self.num_workers)(
            delayed(self.run_episode)(env, render, log_dict)
            for _ in tqdm(range(num_episodes), disable=(num_episodes == 1))
        )
        # results consisting of list of tuples with (eps_duration, rewards, scalar_metrics) for each episode
        # unzip to separate lists with entries for each episode (rewards and metrics are lists of lists; for each step)
        eps_duration, rewards, scalar_metrics, vector_metrics = map(list, zip(*zipped_results))

        # summarize results
        scalar_results = self.summarize_scalar_results(eps_duration, rewards, scalar_metrics)
        mean_results = {metric: np.mean(results) for metric, results in scalar_results.items()}
        self.log.info('Scalar results', results=scalar_results)
        self.log.info('Mean results', results=mean_results)
        self.log.info("Simulation complete", num_episodes=num_episodes, eps_length=self.episode_length,
                      step_reward_mean=np.mean(scalar_results['step_reward_mean']),
                      step_reward_std=np.std(scalar_results['step_reward_std']),
                      avg_eps_reward=self.episode_length * np.mean(scalar_results['step_reward_mean']))

        # write results to file
        if write_results:
            self.write_scalar_results(scalar_results)
            dfs = self.write_vector_results(vector_metrics)

        return rewards
