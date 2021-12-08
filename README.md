# Gaussian Processes on Graphs via Spectral Kernel Learning

This repository contains codes that produced the results in the paper <a href="https://arxiv.org/abs/2006.07361">Gaussian Processes on Graphs via Spectral Kernel Learning</a>.

<code>gma</code> folder contains codes for polynomial spectral kernel learning on synthetic data.

We use the real world datasets *fmri*, *weather*, and *uber*.

<code>main.py</code> implements our proposed model of polynomial spectral kernel learning. For example, the following computes a degree 2 polynomial on *weather* data using 15 signals as training:

```
python main.py --data=weather --training=15 --degree=2 --constrained=on
```
The following training sizes are used in the paper: *fmri*: 21 and 42, *uber*: 10 and 20, *weather*: 15 and 30. <code>--constrained</code> can be set to <code>off</code> to skip the constrained optimization step.

<code>baselines.py</code> computes the performances from the baseline models.

```
python baselines.py --data=weather --training=15 --model=standard
```
Baseline <code>--model</code> options are <code>standard, laplacian, local_averaging, global_filtering, regularized_laplacian, diffusion, 1_random_walk, 3_random_walk, cosine</code>.
