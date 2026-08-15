"""Train a shared multi-agent PPO policy on ``mobile-medium-ma-v0`` with Ray RLlib.

Mirrors the training flow in ``examples/rllib.ipynb`` (see that notebook for an explained,
interactive walkthrough) but runs as a plain script with a longer budget, so it can train to
convergence unattended and produce a checkpoint for ``render_gif.py``.

Usage::

    python docs/scripts/train_rllib.py --num-iters 200 --checkpoint-dir ~/ray_results/mobile-env

Checkpoints are written every ``--checkpoint-every`` iterations (and at the end) under
``--checkpoint-dir``, together with a ``progress.csv`` of the return curve.
"""

import argparse
import csv
import time
from pathlib import Path

import gymnasium
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

import mobile_env  # noqa: F401
from mobile_env.wrappers.multi_agent import RLlibMAWrapper

ENV_NAME = "mobile-medium-ma-v0"
POLICY_ID = "shared_policy"


def register(config):
    env = gymnasium.make(ENV_NAME)
    return RLlibMAWrapper(env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-iters", type=int, default=200, help="max PPO training iterations")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("~/ray_results/mobile-env-medium-ma").expanduser(),
        help="where to write checkpoints and the progress log",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--num-cpus", type=int, default=4, help="total CPUs ray may use")
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=15,
        help=(
            "stop early if no single iteration's return mean beats the running best for this "
            "many iterations in a row; noisy runs (occasional single-iteration spikes) can keep "
            "resetting the counter and prevent this from ever firing -- it's not a smoothed check"
        ),
    )
    args = parser.parse_args()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.checkpoint_dir / "progress.csv"

    register_env(ENV_NAME, register)
    ray.init(num_cpus=args.num_cpus, include_dashboard=False, ignore_reinit_error=True)

    config = (
        PPOConfig()
        .environment(env=ENV_NAME)
        .multi_agent(
            policies={POLICY_ID},
            policy_mapping_fn=lambda agent_id, episode, **kwargs: POLICY_ID,
        )
        # 1 CPU is reserved for the driver/learner; the rest run env rollouts.
        .env_runners(num_env_runners=max(args.num_cpus - 1, 1), num_cpus_per_env_runner=1)
    )
    algo = config.build_algo()

    best_return = float("-inf")
    best_iter = 0
    with open(progress_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iter", "return_mean", "steps_sampled", "seconds"])

        for i in range(1, args.num_iters + 1):
            t0 = time.time()
            result = algo.train()
            dt = time.time() - t0

            return_mean = result.get("env_runners", {}).get("episode_return_mean")
            steps = result.get("num_env_steps_sampled_lifetime")
            writer.writerow([i, return_mean, steps, f"{dt:.1f}"])
            f.flush()
            print(
                f"iter {i}/{args.num_iters}: return_mean={return_mean} ({dt:.1f}s, {steps} steps)"
            )

            if return_mean is not None and return_mean > best_return:
                best_return, best_iter = return_mean, i

            if i % args.checkpoint_every == 0:
                algo.save(str(args.checkpoint_dir / f"checkpoint_{i:04d}"))

            if i - best_iter >= args.plateau_patience:
                print(
                    f"return_mean has not improved on {best_return:.3f} (iter {best_iter}) "
                    f"for {args.plateau_patience} iterations, stopping early"
                )
                break

    final_checkpoint = args.checkpoint_dir / "checkpoint_final"
    algo.save(str(final_checkpoint))
    print(f"saved final checkpoint to {final_checkpoint}")
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
