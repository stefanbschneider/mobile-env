"""Main execution script used for experimentation"""
import os
import logging

import structlog

from deepcomp.util.simulation import Simulation
from deepcomp.util.logs import config_logging
from deepcomp.util.env_setup import create_env_config
from deepcomp.util.cli import setup_cli


log = structlog.get_logger()


def main():
    config_logging()
    args = setup_cli()
    # can't use args.continue: https://stackoverflow.com/a/63266666/2745116
    args_continue = getattr(args, 'continue')

    # stop training when any of the criteria is met
    stop_criteria = {}
    if args.train_steps is not None:
        stop_criteria['timesteps_total'] = args.train_steps
    if args.train_iter is not None:
        stop_criteria['training_iteration'] = args.train_iter
    if args.target_reward is not None:
        stop_criteria['episode_reward_mean'] = args.target_reward
    if args.target_utility is not None:
        stop_criteria['custom_metrics/sum_utility_mean'] = args.target_utility

    # train or load trained agent; only set train=True for ppo agent
    train = args.test is None
    agent_path = None
    if args.test is not None:
        agent_path = os.path.abspath(args.test)
    agent_path_continue = None
    if args_continue is not None:
        agent_path_continue = os.path.abspath(args_continue)

    # create RLlib config (with env inside) & simulator
    config = create_env_config(args)

    # for sequential multi agent env
    # config['no_done_at_end'] = True

    # for continuous training without any resets between episodes
    if args.cont_train:
        config['soft_horizon'] = True
        config['no_done_at_end'] = True

    # default ppo params: https://docs.ray.io/en/latest/rllib-algorithms.html#proximal-policy-optimization-ppo
    # config['entropy_coeff'] = 0.01
    # lr: 5e-5, lr_schedule: None, gae lambda: 1.0, kl_coeff: 0.2
    # config['lr'] = ray.tune.uniform(1e-6, 1e-4)
    # config['gamma'] = ray.tune.uniform(0.9, 0.99)
    # config['lambda'] = ray.tune.uniform(0.7, 1.0)
    # lr_schedule: https://github.com/ray-project/ray/issues/7912#issuecomment-609833914
    # eg, [[0, 0.01], [1000, 0.0001]] will start (t=0) lr=0.01 and linearly decr to lr=0.0001 at t=1000
    # config['lr_schedule'] = [[0, 0.01], [50000, 1e-5]]
    # import hyperopt as hp
    # from ray.tune.suggest.hyperopt import HyperOptSearch
    # hyperopt = HyperOptSearch(metric='episode_reward_mean', mode='max')

    # add cli args to the config for saving inputs
    sim = Simulation(config=config, agent_name=args.alg, cli_args=args, debug=False)

    # train
    if train and args.alg == 'ppo':
        agent_path, analysis = sim.train(stop_criteria, restore_path=agent_path_continue)

    # load & test agent
    sim.load_agent(rllib_dir=agent_path, rand_seed=args.seed, fixed_action=[1, 1], explore=False)

    # simulate one episode and render
    log_dict = {
        'deepcomp.util.simulation': logging.DEBUG,
        # 'deepcomp.env.entities.user': logging.DEBUG,
        # 'deepcomp.env.entities.station': logging.DEBUG
    }
    # set episode randomization for testing and evaluation according to CLI arg
    sim.run(render=args.video, log_dict=log_dict)

    # evaluate over multiple episodes
    if args.eval > 0:
        sim.run(num_episodes=args.eval, write_results=True)

        # evaluate again with toggled episode randomization if --fixed-rand-eval
        if args.fixed_rand_eval:
            log.info('Evaluating again with toggled episode randomization', rand_episodes=not args.rand_test)
            # set changed testing mode which is then saved to the data frame
            sim.cli_args.rand_test = not args.rand_test
            # make new result filename to avoid overwriting the existing one
            sim.set_result_filename()
            sim.run(num_episodes=args.eval, write_results=True)

    log.info('Finished', agent=agent_path)


if __name__ == '__main__':
    main()
