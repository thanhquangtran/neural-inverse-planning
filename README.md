# CS6208 Inverse Planning

This repository contains a gridworld inverse-planning project built around an
exact Bayesian observer and neural approximations of that observer.

The main submission artifact is `demo.ipynb`. It walks through the task,
qualitative exact-inference examples, and three experiment questions:

1. same-layout generalization to held-out trajectories;
2. random-layout and random-goal generalization, including a data-scaling check;
3. rationality ablations under different Boltzmann `beta` values.

## Repository layout

- `demo.ipynb` — executed tutorial notebook and main experiment report.
- `inverse_planning/` — task, planning, inference, dataset, model, training, and visualization code.
- `scripts/` — command-line entrypoints for dataset generation and training.
- `artifacts/` — ignored runtime outputs such as datasets and model checkpoints.

## Setup

The notebook uses optional dependencies for the memoized exact planner,
training, and notebook execution. From a fresh checkout:

```bash
python3 -m pip install -e '.[memo,train,notebook]'
```

If you use the project-local virtual environment from this repo, use:

```bash
.venv/bin/python -m pip install -e '.[memo,train,notebook]'
```

## Run the notebook

To execute the notebook in place and keep the outputs in `demo.ipynb`:

```bash
.venv/bin/python -m jupyter nbconvert \
  --to notebook \
  --execute demo.ipynb \
  --inplace \
  --ExecutePreprocessor.kernel_name=cs6208-venv \
  --ExecutePreprocessor.timeout=-1
```

To write a separate executed copy instead:

```bash
.venv/bin/python -m jupyter nbconvert \
  --to notebook \
  --execute demo.ipynb \
  --output demo.executed.ipynb \
  --ExecutePreprocessor.kernel_name=cs6208-venv \
  --ExecutePreprocessor.timeout=-1
```

## Command-line utilities

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

Datasets and model checkpoints are intentionally ignored by Git because they
are generated artifacts. Re-run the notebook or scripts to regenerate them.
