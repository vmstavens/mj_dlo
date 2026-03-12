
# `mj_dlo`

Hi there,

This small demo project should give you a sneak peek into how to do DLO simulations in MuJoCo.

The project consists of two main files:
1. [sim.py](sim.py)
2. [utils.py](utils.py)

`sim.py` is the main event, where you will find the script you need for simulations.
`utils.py` contains helper functions to build MuJoCo models programmatically.

## Requirements

- Python 3.12+

## Quick Start

```bash
pipx install uv
```

If you do not have `pipx`, you can use `pip install uv` instead ([docs](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
uv sync
```
```bash
uv run python sim.py
```

If you do not want to use `uv`, you can use `pip` instead:

```bash
python3.12 -m venv venv
```
```bash
source venv/bin/activate
```
```bash
pip install -e .
```
```bash
python sim.py
```

You should see a MuJoCo viewer window with a flexible cable attached to a mocap body.

## What to tweak

- In `sim.py`, the `twist`, `bend`, and `size` parameters in `DLOSim._build_model` control the cable stiffness and length.
- In `utils.py`, the `mjx_cable` helper builds the cable using MuJoCo's spec API.

> **OBS**: Be aware that I have two functions `mjs_cable` and `mjx_cable`, `mjs_cable` uses the composite plugin, whereas `mjx_cable` build the cable based on the C code from [MuJoCo](https://github.com/google-deepmind/mujoco/blob/main/src/user/user_composite.cc#L243) using MuJoCo specs. Since `mjx_cable` is not a official MuJoCo cable builder, results may vary.

## Typings

You can use `pybind11-stubgen` to generate Python stubs for `mujoco` into the `typings` folder:

```bash
uv run pybind11-stubgen mujoco -o typings
```
