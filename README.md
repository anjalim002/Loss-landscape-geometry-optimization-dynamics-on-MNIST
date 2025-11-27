# Loss-landscape-geometry-optimization-dynamics-on-MNIST


This repository contains code and example outputs for analyzing loss landscape geometry and optimization dynamics of a neural network trained on MNIST.

The core idea is to go beyond just accuracy and study how the shape of the loss surface (curvature, sharpness, connectivity between solutions) relates to:

*Optimization stability (learning rate vs curvature)

*Speed of convergence

*Generalization performance (test accuracy)

Files in this repository

->mnist_geometry.py
Main script. Trains a simple MLP on MNIST using SGD and periodically logs geometry-related quantities along the training trajectory:

Training loss

Test loss

Test accuracy

Gradient norm

Approximate largest Hessian eigenvalue (λ_max) via power iteration

Directional sharpness (loss increase under small parameter perturbations)

A combined Loss Geometry Difficulty Index (LGDI)

The stability ratio lr * λ_max (to study the edge-of-stability regime)

A 1D loss curve between two checkpoints (simple mode connectivity)

->mnist_geometry_metrics.json 
JSON log containing one entry per probed epoch with:

epoch, train_loss, test_loss, test_accuracy

grad_norm, lambda_max, directional_sharpness

lgdi, stability_ratio

->line_loss.json
JSON log of (alpha, loss) values along a straight line in parameter space between two checkpoints (e.g., epoch 2 and epoch 10). This is used to visualize whether the two minima are connected by a low-loss path or separated by a barrier.
