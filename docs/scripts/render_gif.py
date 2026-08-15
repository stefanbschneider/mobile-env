"""Render episodes of a trained multi-agent policy on ``mobile-medium-ma-v0`` to a GIF.

Loads an RLlib checkpoint written by ``train_rllib.py``, runs it greedily (mode, not sampled)
for one or more episodes, and stitches the ``rgb_array`` frames into an animated GIF using the
env's own ``render_fps`` (see ``MComCore.metadata``) for playback speed, so the encoded GIF and
the live pygame ("human" mode) window are paced the same way.

Usage::

    python docs/scripts/render_gif.py ~/ray_results/mobile-env-medium-ma/checkpoint_final \\
        --out docs/images/mobile-env.gif
"""

import argparse
from pathlib import Path

import gymnasium
import numpy as np
import torch
from PIL import Image
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.columns import Columns
from ray.tune.registry import register_env

import mobile_env  # noqa: F401
from mobile_env.core.base import MComCore
from mobile_env.wrappers.multi_agent import RLlibMAWrapper

ENV_NAME = "mobile-medium-ma-v0"
POLICY_ID = "shared_policy"


def register(config):
    env = gymnasium.make(ENV_NAME)
    return RLlibMAWrapper(env)


def collect_frames(checkpoint_path: str, num_episodes: int, seed: int) -> list[np.ndarray]:
    # Algorithm.from_checkpoint() recreates the algorithm's env runners from the checkpointed
    # config, which needs the env registered under its name again, just like train_rllib.py.
    register_env(ENV_NAME, register)
    algo = Algorithm.from_checkpoint(checkpoint_path)
    module = algo.get_module(POLICY_ID)
    action_dist_cls = module.get_inference_action_dist_cls()

    env = gymnasium.make(ENV_NAME, render_mode="rgb_array")
    frames = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        while not done:
            action = {}
            for agent_id, agent_obs in obs.items():
                input_dict = {
                    Columns.OBS: torch.from_numpy(np.array([agent_obs], dtype=np.float32))
                }
                logits = module.forward_inference(input_dict)[Columns.ACTION_DIST_INPUTS]
                # greedy (mode) action for a clean, reproducible GIF instead of a sampled one
                action[agent_id] = int(
                    action_dist_cls.from_logits(logits).to_deterministic().sample()[0]
                )

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            frames.append(env.render())

    env.close()
    algo.stop()
    return frames


def save_gif(frames: list[np.ndarray], out_path: Path, fps: int, scale: float, colors: int) -> None:
    images = [Image.fromarray(frame) for frame in frames]
    if scale != 1.0:
        size = (round(images[0].width * scale), round(images[0].height * scale))
        images = [img.resize(size, Image.LANCZOS) for img in images]

    # quantize to a shared, size-limited palette; GIF is palette-based anyway, and mobile-env's
    # renders use few distinct colors, so this keeps the file small with little visible loss
    palette = images[0].quantize(colors=colors, dither=Image.FLOYDSTEINBERG)
    images = [
        img.quantize(colors=colors, palette=palette, dither=Image.FLOYDSTEINBERG) for img in images
    ]

    duration_ms = round(1000 / fps)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=str, help="path to an RLlib checkpoint directory")
    parser.add_argument("--out", type=Path, default=Path("docs/images/mobile-env.gif"))
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scale", type=float, default=0.75, help="resize factor to keep the GIF small"
    )
    parser.add_argument(
        "--colors", type=int, default=128, help="palette size to keep the GIF small"
    )
    args = parser.parse_args()

    frames = collect_frames(args.checkpoint, args.episodes, args.seed)
    fps = MComCore.metadata["render_fps"]
    save_gif(frames, args.out, fps=fps, scale=args.scale, colors=args.colors)
    print(f"wrote {len(frames)} frames to {args.out} at {fps} fps")


if __name__ == "__main__":
    main()
