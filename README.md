# CS6208 Inverse Planning

This repository contains a simple gridworld inverse-planning project built
around an exact Bayesian baseline and reusable RNN approximations.

## Contents

- `demo.ipynb`: notebook demo with task overview, exact inference, and visuals.
- `inverse_planning/`: reusable package for task setup, planning, inference, data, models, and visualization.
- `scripts/`: dataset generation and training entrypoints.
- `latex/`: starter project report and bibliography.

## Quick Start

```bash
python3 -m pip install -e .
python3 scripts/demo_workflow.py
```

Optional training dependencies:

```bash
python3 -m pip install -e .[train]
```

Optional LaTeX report tooling:

```bash
uv pip install --python .venv/bin/python b8tex
.venv/bin/python -m pip install -e .[latex]
.venv/bin/inverse-planning-compile-latex
```

## Project Idea

The current implementation uses the simple `inv_plan_from_scratch` task as a
tractable reference setting. Exact inverse planning provides ground-truth
posteriors over goals, and RNN variants are trained either to predict goals
directly or to score actions under candidate goals.
