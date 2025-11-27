# Loss Landscape Geometry & Optimization Dynamics

This repository contains code and example outputs for analyzing the **loss landscape geometry** and **optimization dynamics** of a neural network trained on MNIST.

Instead of looking only at accuracy, the goal here is to study how the **shape of the loss surface** (curvature, sharpness, connectivity between solutions) relates to:

- Optimization stability (learning rate vs curvature)  
- Speed and behaviour of SGD during training  
- Generalization performance (test accuracy)  

This directly targets the problem:

> *“Develop a rigorous framework for analyzing neural network loss landscape geometry and its relationship to optimization dynamics, generalization, and architecture design. Derive theoretical results, implement efficient landscape probing methods, and empirically validate connections between geometric properties and model behavior.”*

---

## Files in this repository

### 1. `mnist_geometry.py`

Main script. It:

- Builds a small **MLP** for MNIST.
- Trains the model using **SGD with momentum**.
- Periodically probes the loss landscape and logs:

  - `train_loss` – training loss  
  - `test_loss` – test loss  
  - `test_accuracy` – test accuracy  
  - `grad_norm` – ℓ₂ norm of the gradient on a mini-batch  
  - `lambda_max` – approximate largest **Hessian eigenvalue** (local curvature/sharpness) via power iteration  
  - `directional_sharpness` – average loss increase along small random parameter perturbations  
  - `lgdi` – **Loss Geometry Difficulty Index**, combining learning rate, curvature, gradient norm and sharpness  
  - `stability_ratio` – `lr * lambda_max` (to study the **edge-of-stability** regime)

- At the end, it also computes a **1D loss curve between two checkpoints** (e.g., epoch 2 and epoch 10):

  - For `alpha ∈ [0, 1]`, evaluates the loss at  
    \[
      \theta(\alpha) = (1 - \alpha)\,\theta_{\text{early}} + \alpha\,\theta_{\text{late}}
    \]
  - This is used to study **mode connectivity** along a straight line in parameter space.

---

### 2. `mnist_geometry_metrics.json`

Example output file produced by running `mnist_geometry.py`.

It is a JSON list where each element corresponds to one “geometry probe” (one epoch where metrics were measured), containing keys like:

- `"epoch"`  
- `"train_loss"`, `"test_loss"`, `"test_accuracy"`  
- `"grad_norm"`  
- `"lambda_max"`  
- `"directional_sharpness"`  
- `"lgdi"`  
- `"stability_ratio"`

You can use this file in a notebook to plot:

- Train vs test loss  
- Test accuracy vs epoch  
- Curvature (λ_max) vs epoch  
- Stability ratio (`lr * λ_max`) vs epoch  
- LGDI vs epoch  

---

### 3. `line_loss.json`

Another example output file produced by running `mnist_geometry.py`.

It is a JSON list of objects with:

- `"alpha"` – interpolation parameter between two checkpoints (0 = early, 1 = late)  
- `"loss"` – loss at the interpolated parameters

This describes the **loss along a straight line** between two parameter vectors. A relatively flat curve suggests the two solutions lie in a connected low-loss valley; a pronounced “bump” suggests a barrier along that line.
