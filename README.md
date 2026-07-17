# Pass/Fail Predictor — Neural Network from Scratch in NumPy

A 2-layer neural network built entirely from scratch using raw NumPy — no PyTorch, no TensorFlow, no autograd. Forward propagation, backpropagation, and gradient descent are all implemented manually, in order to understand exactly what deep learning frameworks are doing under the hood before relying on them.

## Problem

Toy binary classification task: predict whether a student passes or fails based on two features — hours studied and attendance percentage.

## Why this project exists

Before using PyTorch's `autograd` and built-in layers, I wanted to implement the underlying math by hand: the forward pass, the derivatives for backpropagation, and the parameter update rule — to make sure I understood *why* the framework abstractions work, not just how to call them.

## Architecture

- Input layer: 2 features (study hours, attendance %)
- Hidden layer: 16 units, `tanh` activation
- Output layer: 1 unit, `sigmoid` activation (binary classification)
- Loss: binary cross-entropy
- Optimizer: manual gradient descent (no momentum, no Adam — just the raw update rule)

## Implementation notes

- **Feature normalization:** the two input features are on very different scales (study hours ~0.5–5, attendance ~30–90%), which slows convergence and can cause instability if left unnormalized. Both features are standardized (zero mean, unit variance) before training.
- **Weight initialization:** weights are initialized small (`* 0.01`) rather than raw `np.random.randn()`, which helps avoid saturating the `tanh`/`sigmoid` activations early in training.
- **Reproducibility:** a fixed random seed (`np.random.seed(42)`) ensures consistent results across runs.

## Results

Training converges cleanly, with loss dropping from ~0.69 (random guessing) to ~0.0002 over 10,000 iterations, reaching 100% accuracy on the training set.

```
Step 0,    Loss: 0.6931, Accuracy: 0.38
Step 1000, Loss: 0.0031, Accuracy: 1.00
...
Final training accuracy: 1.00
```

## An honest limitation

This dataset has only 8 students and no train/test split — with 16 hidden units, the network has more than enough capacity to simply memorize these 8 exact examples rather than learn a generalizable pattern. This project is meant to demonstrate correct implementation of the underlying math (forward/backward propagation), not as a real predictive model. A genuine version of this task would need a much larger, held-out dataset to evaluate actual generalization.

## What I'd do to make this a real model

- Collect or find a real, larger dataset of student performance
- Add a train/test split to measure generalization instead of memorization
- Add L2 regularization or dropout to prevent overfitting on a larger dataset
- Compare against the equivalent model built in PyTorch, to confirm the manual implementation produces the same results as the framework's autograd

## Tech stack

Python, NumPy

## How to run

```bash
python3 pass_fail_predictor.py
```
