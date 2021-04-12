import argparse

import structlog

from deepcomp.util.constants import SUPPORTED_ALGS, SUPPORTED_ENVS, SUPPORTED_AGENTS, SUPPORTED_RENDER, \
    SUPPORTED_SHARING, SUPPORTED_REWARDS, CENTRAL_ALGS, MULTI_ALGS


log = structlog.get_logger()


def setup_cli():
    """Create CLI parser and return parsed args"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # algorithm & training
    parser.add_argument('--approach', type=str, choices=['deepcomp', 'ddcomp', 'd3comp'], default=None,
                        help="Which DRL approach to use: DeepCoMP, DD-CoMP, or D3-CoMP. "
                             "Overrides --agent, --alg, and --separate-agent-nns.")
    parser.add_argument('--agent', type=str, choices=SUPPORTED_AGENTS, default='central',
                        help="Whether to use a single agent for 1 UE, a central agent, or multi agents")
    parser.add_argument('--alg', type=str, choices=SUPPORTED_ALGS, default='ppo', help="Algorithm")
    parser.add_argument('--workers', type=int, default=1, help="Number of workers for training (one per CPU core)")
    parser.add_argument('--batch-size', type=int, default=4000, help="Number of training iterations per training batch")
    parser.add_argument('--train-steps', type=int, default=None, help="Max. number of training time steps (if any)")
    parser.add_argument('--train-iter', type=int, default=None, help="Max. number of training iterations (if any)")
    parser.add_argument('--target-reward', type=int, default=None, help="Target mean episode reward for training")
    parser.add_argument('--target-utility', type=int, default=None, help="Target mean sum utility for training")
    parser.add_argument('--continue', type=str, help="Continue training agent at given path (loads last checkpoint")
    parser.add_argument('--separate-agent-nns', action='store_true',
                        help="Only relevant for multi-agent RL. Use separate NNs for each agent instead of sharing.")
    parser.add_argument('--lstm', action='store_true', help="Whether or not to use an LSTM cell")
    parser.add_argument('--reward', type=str, choices=SUPPORTED_REWARDS, default='sum',
                        help="How to aggregate rewards from multiple UEs within a step.")
    parser.add_argument('--cluster', action='store_true', help="Set this flag when running on a multi-node cluster.")
    # environment
    parser.add_argument('--env', type=str, choices=SUPPORTED_ENVS, default='custom', help="Env/Map size")
    parser.add_argument('--num-bs', type=int, default=None, help="Number of BS in large env (not supported by others).")
    parser.add_argument('--bs-dist', type=int, default=100, help="Distance between BS. Only supported by medium env.")
    parser.add_argument('--eps-length', type=int, default=100, help="Number of time steps per episode")
    parser.add_argument('--max-ues', type=int, default=None, help="Expected max. number of UEs. Relevant for central "
                                                                  "agent's NN size. Derived automatically if not set.")
    parser.add_argument('--static-ues', type=int, default=0, help="Number of static UEs in the environment")
    parser.add_argument('--ues', type=int, default=0, help="Number of (slow) UEs in the environment")
    parser.add_argument('--fast-ues', type=int, default=0, help="Number of fast UEs in the environment")
    # could implement this simply by processing the number and increasing the static, slow, fast UEs correspondingly
    # before passing to env creation
    # parser.add_argument('--mixed-ues', type=int, default=0,
    #                     help="Number of UEs in the environment, equally mixed between different movement speeds")
    parser.add_argument('--new-ue-interval', type=int, default=None,
                        help="Interval in number of steps after which a new UEs enter the environment periodically.")
    parser.add_argument('--sharing', type=str, choices=SUPPORTED_SHARING.union({'mixed'}), default='mixed',
                        help="Sharing model used by BS to split resources and/or rate among connected UEs.")
    # evaluation
    parser.add_argument('--rand-train', action='store_true', help="Randomize training episodes.")
    parser.add_argument('--cont-train', action='store_true', help="Continuous training without resetting.")
    parser.add_argument('--rand-test', action='store_true', help="Randomize testing and evaluation episodes.")
    parser.add_argument('--fixed-rand-eval', action='store_true',
                        help="Evaluate once with fixed episodes and then again with random episodes.")
    parser.add_argument('--test', type=str, help="Test trained agent at given path (auto. loads last checkpoint)")
    parser.add_argument('--video', type=str, choices=SUPPORTED_RENDER, default='html',
                        help="How (and whether) to render the testing video.")
    parser.add_argument('--dashboard', action='store_true', help="Render video in form of a dashboard (slow).")
    parser.add_argument('--ue-details', action='store_true', default=True, help="Show UEs' data rate and util in rendered video.")
    parser.add_argument('--eval', type=int, default=0, help="Number of evaluation episodes after testing")
    parser.add_argument('--seed', type=int, default=42, help="Seed for the RNG (algorithms and environment)")
    parser.add_argument('--result-dir', type=str, default=None, help="Optional path to where results should be stored."
                                                                     "Default: <project_root>/results")
    # TODO: changed defaults for demo: seed was None, env was small, --ues was --slow-ues; --ue-details default true

    args = parser.parse_args()

    # TODO: for demo --slow-ues --> --ues
    args.slow_ues = args.ues

    # shortcut cli command for deepcomp, ddcomp, d3comp
    if args.approach is not None:
        args.alg = 'ppo'
        if args.approach == 'deepcomp':
            args.agent = 'central'
            args.separate_agent_nns = False
        elif args.approach == 'ddcomp':
            args.agent = 'multi'
            args.separate_agent_nns = False
        elif args.approach == 'd3comp':
            args.agent = 'multi'
            args.separate_agent_nns = True

    # check if algorithm and agent are compatible or adjust automatically
    if args.alg in CENTRAL_ALGS and args.alg not in MULTI_ALGS and args.agent != 'central':
        log.warning('Algorithm only supports central agent. Switching to central agent.', alg=args.alg)
        args.agent = 'central'
    if args.alg in MULTI_ALGS and args.alg not in CENTRAL_ALGS and args.agent != 'multi':
        log.warning('Algorithm only supports multi-agent. Switching to multi-agent.', alg=args.alg)
        args.agent = 'multi'

    # render settings
    if (args.dashboard or args.ue_details) and args.video is None:
        log.warning("--dashboard and --ue-details have no effect without --video.")

    log.info('CLI args', args=args)

    assert getattr(args, 'continue') is None or args.test is None, "Use either --continue or --test, not both."
    assert args.rand_test is False or args.fixed_rand_eval is False, "Use either --rand-test or --fixed-rand-eval."
    assert not (args.cont_train and args.rand_train), "Either train continuously or with random episodes."
    if args.num_bs is not None:
        assert args.env == 'large', "--num-bs only supported by large env"
        assert 1 <= args.num_bs <= 7, "--num-bs must be between 1 and 7"
    return args
