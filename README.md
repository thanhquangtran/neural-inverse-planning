# Neural Inverse Planning

This repository studies **inverse planning in a gridworld**: given an agent's
observed states and actions, can we infer which goal it is pursuing?

The project combines:
- an **exact Bayesian observer** built from a Boltzmann-rational policy,
- **neural models** trained to approximate that observer, and
- a **tutorial notebook** that walks through the task, qualitative examples,
  and the main experiments.

## Main artifact

The main artifact is the notebook currently named `neural_inverse_planning_tutorial.ipynb`.
It serves as both:
- a tutorial-style walkthrough of the gridworld inverse-planning setup, and
- the main experiment report for the project.

The notebook is organized around three questions:
1. **Held-out trajectories:** can the models generalize within the same layout?
2. **Random layouts:** can the models generalize across new goal layouts, and does more data help?
3. **Rationality ablation:** how does the agent's Boltzmann rationality affect inference difficulty?

## Repository structure

- `neural_inverse_planning_tutorial.ipynb` — tutorial notebook and main experiment walkthrough
- `inverse_planning/` — task definition, exact inference, simulation, visualization, models, and training code
- `scripts/` — command-line utilities for dataset generation and training
- `artifacts/` — generated datasets, checkpoints, and intermediate experiment outputs (gitignored)
- `latex/` — report and slide materials

## Setup

Install the optional dependencies needed for the exact planner, training, and notebook workflow:

```bash
python3 -m pip install -e '.[memo,train,notebook]'
```

If you use the project-local virtual environment:

```bash
.venv/bin/python -m pip install -e '.[memo,train,notebook]'
```

## Using the project

### Notebook
Open `neural_inverse_planning_tutorial.ipynb` in Jupyter and run it as the main walkthrough of the project.

### Command-line utilities
Generate a dataset:

```bash
inverse-planning-generate-dataset --output artifacts/train.npz --episodes 1024
```

Train a classifier model:

```bash
inverse-planning-train-goal-belief-rnn \
  --dataset artifacts/train.npz \
  --variant conv_gru \
  --output artifacts/conv_gru.pt
```

Train the goal-conditioned policy model:

```bash
inverse-planning-train-goal-conditioned-policy \
  --dataset artifacts/train.npz \
  --output artifacts/goal_conditioned_policy.pt
```

Run the small smoke-test workflow:

```bash
inverse-planning-demo
```

## Notes

- The exact observer uses a **Boltzmann-rational policy** and Bayesian goal inference as the reference target.
- Neural models are trained to approximate that inference process from trajectory data.
- Datasets and checkpoints are generated artifacts and are intentionally not tracked in Git.
