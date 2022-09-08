# Gaussian Processes on Graphs via Spectral Kernel Learning

This repository contains codes that produced the results in the paper **Gaussian Processes on Graphs via Spectral Kernel Learning**
<!--
<a href="https://arxiv.org/abs/2006.07361" target="_blank">Gaussian Processes on Graphs via Spectral Kernel Learning</a>.
-->

We recommend starting with the <code>gma</code> folder which contains codes for learning on synthetic data.

The real world datasets used in the paper are *fmri*, *weather*, and *uber*. The following training sizes are used in the paper: 
- *fmri*: 21 and 42
- *uber*: 10 and 20
- *weather*: 15 and 30.

<code>main.py</code> implements our proposed model of polynomial spectral kernel learning. The following computes a degree 2 polynomial on *weather* data using 15 signals as training (set <code>--constrained</code> to <code>off</code> to skip the constrained optimization step):

```
python main.py --data=weather --training=15 --degree=2 --constrained=on
```
<code>baselines.py</code> computes the performances from the baseline models, a <code>--model</code> term needs to be specified instead of <code>--degree</code> and <code>--constrained</code>:

```
python baselines.py --data=weather --training=15 --model=standard
```
Baseline <code>--model</code> options are <code>standard</code>, <code>laplacian</code>, <code>local_averaging</code>, <code>global_filtering</code>, <code>regularized_laplacian</code>, <code>diffusion</code>, <code>1_random_walk</code>, <code>3_random_walk</code>, <code>cosine</code>.
