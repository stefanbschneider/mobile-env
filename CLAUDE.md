# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Setup (editable install + test/lint deps):

```bash
pip install -e .
pip install -r tests/requirements.txt
```

Test / lint (mirrors `.github/workflows/python-package.yml`):

```bash
pytest                                                       # full suite
pytest tests/test_env_stepping.py -k "small and central"     # single parametrized case
pytest tests/test_central_envs.py::TestCentralEnvs::test_central_small
pytest --nbmake examples/test.ipynb                          # notebooks are part of CI
ruff check .                                                  # lint, exact CI invocation
ruff format --check .                                         # format check, exact CI invocation
pre-commit run --all-files                                   # ruff (lint + format), yaml checks
```

Docs (Sphinx, published to ReadTheDocs via `.readthedocs.yaml`):

```bash
pip install -r docs/requirements.txt
cd docs && make html          # output in docs/_build/html
```

`docs/requirements.txt` pins old versions (sphinx 3.5.4, myst-parser 0.15.2) and RTD builds on Python
3.8, so a local build may need its own virtualenv. `docs/source/*.rst` are checked-in sphinx-apidoc
stubs, not generated at build time — a new module needs a matching rst entry (note that
`mobile_env.wrappers` is currently missing from `docs/source/mobile_env.rst`).

ruff (lint + format) config is in `ruff.toml` (line-length 100, select `E`, `F`, `W`, `I`); there is
no `pyproject.toml`. Both `ruff check` and `ruff format --check` run in CI and via pre-commit.
Python >= 3.10; CI matrix is 3.10–3.13 on ubuntu/macos/windows (see `.github/workflows/python-package.yml`
for the windows+3.13 exclusion).

## Architecture

### Environment registration

`import mobile_env` → `mobile_env/scenarios/__init__.py` → `scenarios/registry.py`, which registers the
cross-product of 3 scenarios (`small`/`medium`/`large`) × 2 handlers (`central`/`ma`) as
`mobile-{scenario}-{handler}-v0`. Registration is a pure import side effect, which is why tests import
`mobile_env` with `# noqa: F401`. A scenario subclass (`scenarios/*.py`) only fixes the map size, BS
positions, and UE count; everything else comes from `MComCore`.

### Strategy-pattern configuration

`MComCore.default_config()` returns a dict whose values for `arrival`, `channel`, `scheduler`,
`movement`, `utility`, and `handler` are **classes**, each paired with a `<name>_params` dict passed to
its constructor. User config is merged over the defaults with `deep_dict_merge`, then `seeding()`
assigns each component `seed + n + 1` so components get distinct but deterministic RNGs.

Extending any simulation aspect therefore means: subclass the base class in `mobile_env/core/`
(`Channel`, `Arrival`, `Movement`, `Scheduler`, `Utility`), then set both `config['<name>']` and
`config['<name>_params']` — no source changes needed. See `docs/components.md`.

Each component exposes `reset()` plus its own abstract methods; components that own RNG state honour
`reset_rng_episode` (default `False`, i.e. randomness continues across episodes rather than repeating).

### Handlers are the Gym seam

`MComCore` itself is agnostic to the RL interface. `MComCore.features()` computes a **superset** of
per-UE features (`connections`, `snrs`, `utility`, `bcast`, `stations_connected`; lengths declared in
`self.feature_sizes`). Each `Handler` (`mobile_env/handlers/`) declares a `features` class attribute
selecting a subset, and owns `action_space`, `observation_space`, `action()` (reshaping into the
`{ue_id: action}` dict the core expects), `observation()`, `reward()`, `check()`, and `info()`.

- `MComCentralHandler`: `MultiDiscrete` action, one flat `Box` observation concatenated over all UEs,
  reward = mean scaled utility.
- `MComMAHandler`: `Dict` spaces keyed by `ue_id`, observations/rewards only for currently active UEs.

Adding an observation feature requires touching three places: `features()`, `feature_sizes`, and the
handler's `features` list.

### `step()` semantics worth knowing

- `terminated` is always `False`; `truncated` is `time_is_up`, i.e. `time >= min(EP_MAX_TIME, max_departure)`.
- Data rates are recomputed twice per step: after applying actions, and again after UEs move.
- `if not self.active and not self.time_is_up: return self.step({})` — a single `env.step()` call can
  advance several simulation time steps when no UE is requesting service.
- Utilities are scaled to `[-1, 1]` before rewards are computed; `render()` unscales them again.

### Monitoring

`config['metrics']` holds `scalar_metrics` / `ue_metrics` / `bs_metrics` dicts of `callable(sim)`.
`MComCore.__init__` injects the four scalar metrics that `render()` depends on. The `info` dict
returned by `reset()`/`step()` is the handler's info merged with the monitor's latest values;
`Monitor.load_results()` returns the full episode history as DataFrames.

## Gotchas

- `MComCentralHandler.check()` asserts a list comprehension (always truthy), so the "central env cannot
  handle a changing number of UEs" constraint is **not** actually enforced. A custom `Arrival` with real
  departures is only meaningful with the MA handler.
- `RateFair` in `core/schedules.py` is broken as a drop-in replacement for the default `ResourceFair`:
  `share()` returns a scalar instead of a per-UE list (`station_allocation` zips over it), and it
  divides by zero for any connected UE whose data rate is `0.0` (what `Channel.datarate` returns below
  the SNR threshold).
- `README.md` and `docs/components.md` both show `from mobile_env.core.channel import Channel`; the
  module is `mobile_env/core/channels.py`. The README snippet fails verbatim.
- `mobile_env/wrappers/multi_agent.py` imports `ray.rllib` at module top, but ray is commented out of
  `tests/requirements.txt` (Ray >= 2.39 requires gymnasium 1.0, which is not yet supported). The RLlib
  wrapper is therefore untested by CI and needs a separate ray install. `PettingZooWrapper` is a stub.
- `gymnasium` is unpinned in `setup.py` as of 2.1.0 (the earlier `<1.0.0` pin was lifted), but
  stable-baselines3 support constrains what actually works — see `tests/requirements.txt`.
- The package version lives only in `setup.py`.
