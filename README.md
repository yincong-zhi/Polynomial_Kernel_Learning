# Gaussian Processes on Graphs via Spectral Kernel Learning

This repository contains codes that produced the results in the paper [Gaussian Processes on Graphs via Spectral Kernel Learning](https://ieeexplore.ieee.org/document/10093993)
<!--
<a href="https://arxiv.org/abs/2006.07361" target="_blank">Gaussian Processes on Graphs via Spectral Kernel Learning</a>.
-->
# 
We recommend starting with the <code>gma</code> folder which contains codes for learning on synthetic data.

<code>main.py</code> implements our proposed model of polynomial spectral kernel learning. The example command below computes a degree 2 polynomial on the *weather* data using 15 signals as training
```
python main.py --data=weather --training=15 --degree=2 --constrained=on
```
The real world datasets used in the paper are *fmri*, *weather*, and *uber*. For each dataset, the following training sizes are used in the paper:
- <code>fmri</code>: 21 and 42
- <code>uber</code>: 10 and 20
- <code>weather</code>: 15 and 30.

The <code>--constrained</code> parameter can be specified to <code>off</code> to skip constrained optimization.

---
<code>baselines.py</code> computes the performances from the baseline models, a <code>--model</code> term needs to be specified instead of <code>--degree</code> and <code>--constrained</code>:

```
python baselines.py --data=weather --training=15 --model=standard
```
Baseline <code>--model</code> options are:
- <code>standard</code>
- <code>laplacian</code>
- <code>local_averaging</code>
- <code>global_filtering</code>
- <code>regularized_laplacian</code>
- <code>diffusion</code>
- <code>1_random_walk</code>
- <code>3_random_walk</code>
- <code>cosine</code>.
