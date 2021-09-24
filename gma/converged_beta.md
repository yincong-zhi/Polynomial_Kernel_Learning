# Degree 5 ground truth, identity temporal kernel

*theta = [0., 1., 4., 1., -6.]*

degree 2
```
beta2 = [1.62951594e-04, 1.98523826e+00]
log-likelihood = [[-307.08966448]]
```
degree 3
```
beta3 = [ 3.33129493e-07,  3.97399525e+00, -3.30191729e+00]
log-likelihood = [[-272.16061707]]
```
degree 4
```
beta4 = [ 0.01231544,  0.76424834,  7.86562624, -8.57998912]
log-likelihood = [[-201.82558506]] (unconstrained)
beta = [-4.00153974e-05, -1.52939917e-01,  1.01600579e+01, -9.99298578e+00]
log-likelihood = [[-214.60962212]] (constrained)
```
degree 5
```
beta5 = [-0.00740217,  1.4898351,   3.11069651,  0.31830887, -4.88088413]
log-likelihood = [[-199.06508327]] (unconstrained)
beta = [ 6.17708944e-06,  7.23150438e-01,  4.42090549e+00,  4.84764006e-01, -5.62883342e+00]
log-likelihood = [[-205.89717143]] (constrained)
```

# degree 5 ground truth (low pass)

*theta = np.array([1., -1.5, (1.5**2)/2., -(1.5**3)/6., (1.5**4)/24.])*

degree 2
```
beta2 = [ 0.83877209, -0.63230945]
log-likelihood = [[163.14547251]]
```
degree 3
```
beta3 = [ 1.08473474, -1.64230425,  0.83725048]
log-likelihood = [[172.37190434]]
```
degree 4
```
beta4 = [ 0.93037959, -0.40114082, -1.69707867,  1.49283041]
log-likelihood = [[171.39497296]]
```
degree 5
```
beta5 = [ 0.97370473, -0.8594714,  -0.58577507,  0.64472898,  0.13096232]
log-likelihood = [[172.23603843]]
```

# wishart temporal kernel band pass
```
theta = np.array([0., 1., 4., 1., -6.])

beta1 = np.array([0.00437127, 0.87479639])
beta2 = np.array([ 1.83737737e-06,  1.80340144e+00, -1.48285339e+00])
beta3 = np.array([ 3.67878587e-03,  6.97119390e-02,  4.32094669e+00, -4.39479948e+00])
beta4 = np.array([ 0.020502  , -0.41551435,  6.54117888, -7.62595605,  1.47908343])

log_likelihoods = [61.06635996, 81.73771126, 110.91706497, 110.12449929]
```
# low pass
```
theta = np.array([1., -1.5, (1.5**2)/2., -(1.5**3)/6., (1.5**4)/24.])

beta1 = np.array([ 0.72411302, -0.68227031])
beta2 = np.array([ 0.829204  , -1.18790397,  0.47544722])
beta3 = np.array([ 0.7352795 , -0.08747404, -2.19381137,  1.72266287])
beta4 = np.array([ 0.7570524 , -0.5089113 , -0.46219207, -0.80667465,  1.20875661])

log_likelihoods = [645.30978327, 695.17195797, 695.42438226, 695.03693426]
```
---

# varying number of signals

*theta = [0., 1., 4., 1., -6.]*

*degree = 4*

*0.2 noise*

5 signals
```
beta5 = [-3.17191278e-04, -2.24577220e-01,  1.02728520e+01, -1.00478607e+01]
log-likelihood = [[-81.65341767]]
average log-likelihood = [[-16.33068353]]
```
10 signals
```
beta10 = [-5.75607393e-06, -1.80208227e-01,  1.03150039e+01, -1.00834657e+01]
log-likelihood = [[-181.26006008]]
average log-likelihood = [[-18.12600601]]
```
15 signals
```
beta15 = [-2.68637840e-04, -7.56237545e-02,  1.01480595e+01, -1.00320951e+01]
log-likelihood = [[-279.67158142]]
average log-likelihood = [[-18.64477209]]
```
20 signals
```
beta20 = [-1.48211821e-03, -3.10019347e-01,  1.05546482e+01, -1.01779997e+01]
log-likelihood = [[-357.7831714]]
average log-likelihood = [[-17.88915857]]
```
25 signals
```
beta25 = [ 2.63956560e-03, -5.01331523e-01,  1.01979799e+01, -9.67943058e+00]      
log-likelihood = [[-444.00520036]]
average log-likelihood = [[-17.76020801]]
```
30 signals
```
beta30 = [ 4.45887322e-03, -4.41731325e-01,  1.01462253e+01, -9.70885261e+00]      
log-likelihood = [[-537.01250857]]
average log-likelihood = [[-17.90041695]]
```

# varying noise snr
*theta = [0., 1., 4., 1., -6.]*

*degree = 4*

*5 signals*
SNR 2
```
beta5 = [-4.27081877e-05,  1.32831020e+00,  6.39435791e+00, -7.72161074e+00]
noise std = 0.5404283384259578
log-likelihood = [[-141.9931817]]
average log-likelihood = [[-28.39863634]]
```
SNR 2.5
```
beta5 = [  0.02748629,  -0.17090337,  10.21021473, -10.06668885]
noise std = 0.4365189420432045
log-likelihood = [[-122.05578095]]
average log-likelihood = [[-24.41115619]]
```
*10 signals*
SNR 2
```
beta10 = [ 3.97326574e-04,  1.07516527e+00,  6.60880739e+00, -7.35956550e+00]
noise std = 0.5458269934516481
log-likelihood = [[-301.78478879]]
average log-likelihood = [[-30.17847888]]
```
SNR 2.5
```
beta10 = [-7.33190639e-03, -2.81647419e-02,  1.00984244e+01, -9.91288411e+00]
noise std = 0.46183409428811706
log-likelihood = [[-263.9658834]]
average log-likelihood = [[-26.39658834]]
```
*15 signals*
SNR 2
```
beta15 = [-7.32620126e-05,  2.53725199e+00,  1.99882541e+00, -4.03113546e+00]
noise std = 0.552190511324108
log-likelihood = [[-464.97607506]]
average log-likelihood = [[-30.998405]]
```
SNR 2.5
```
beta15 = [ 1.26692008e-03,  1.06672841e+00,  6.56290980e+00, -7.36230755e+00]
noise std = 0.4763922236693263
log-likelihood = [[-408.01231984]]
average log-likelihood = [[-27.20082132]]
```
*20 signals*
SNR 2
```
beta20 = [ 5.72697032e-04,  1.07305003e+00,  6.70733621e+00, -7.56366845e+00]
noise std = 0.5858117330330228
log-likelihood = [[-610.73204364]]
average log-likelihood = [[-30.53660218]]
```

---
*theta = [0., 1., 4., 1., -6.]*
*degree = 4*

10 signals
```
beta = [ 0.03551715,  1.36398584,  4.25607301, -5.34668566]
noise std = 0.4209791379328887
log-likelihood = [[-83.27733845]]
average log-likelihood = [[-8.32773385]]
```
20 signals
```
beta = [ 1.04601657e-03  5.62826124e-01  6.39255218e+00 -6.76785544e+00]
noise std = 0.5777879991432845
log-likelihood = [[-223.66897438]]
average log-likelihood = [[-11.18344872]]
```
40 signals
```
beta = [ 3.00816391e-04  4.40383619e-01  6.63115959e+00 -6.87361911e+00]
noise std = 0.6127299920005262
log-likelihood = [[-486.25009399]]
average log-likelihood = [[-12.15625235]]
```
60 signals
```
beta = [ 0.14448119 -0.63676779  8.89114311 -8.19486617]
noise std = 0.5747081914453588
log-likelihood = [[-748.62765534]]
average log-likelihood = [[-12.47712759]]
```

# different SNR

## band pass

SNR 20
```
unconstrained (already psd)
beta20 = [ 0.10220834, -0.37231369, 10.00538925, -9.64939313]
noise = 0.08309335744982531
log-likelihood = [[-42.722709449999996]]

constrained
beta = [ 0.12704299, -0.45303971,  9.95387496, -9.515869  ]
noise = 0.04140752567573202
constrained log-likelihood = [[-42.60794009999999]]
```
SNR 15
```
unconstrained (already psd)
beta15 = [ 0.06390294, -0.42180212, 10.07582423, -9.64491351]
noise std = 0.2139526254291511
average log-likelihood = [[-70.9865526]]

constrained
beta = [ 0.13878527, -0.68000054, 10.12827052, -9.46216349]
noise std = 0.179972308087377
average log-likelihood = [[-70.8233148]]
```
SNR 10
'''
beta10 = [ 7.98879099e-03, -5.64655631e-01,  1.01750897e+01, -9.51580429e+00]
noise = 0.37511789284894864
constrained log-likelihood = [[-106.94766345]]
'''
SNR 5
```
beta5 = [ 0.54512022, -0.74430183,  6.50101952, -5.72719974]
noise = 2.7281001263891874e-12
constrained log-likelihood = [[-163.60123335]]
```
SNR 0
```
beta0 = [ 0.76432204, -0.71103246,  4.40718765, -3.6789571 ]
noise std = 7.022387972415122e-06
constrained log-likelihood = [[-248.4076188]]
```

# low pass

SNR 20
beta20 = [ 1.9720012,  -3.20429148,  1.77277672]
noise = 0.01790596683108631
marginal = [[-86.02080911]]

SNR 15
beta15 = [ 1.9634596,  -3.19422297,  1.78876263] 
noise = 0.026047359365406104
marginal = [[-92.25371556]]

SNR 10
beta10 = [ 1.94036244, -3.16177553,  1.83012245] 
noise = 0.06445035550930915
marginal = [[-111.07336503]] 

SNR 5
beta5 = [ 1.89219502, -3.0843399,   1.92696635]
noise = 0.07687751719023857
marginal = [[-157.68431203]]

SNR 0
beta0 = [ 1.63263572, -2.89332948,  1.93948902]
noise = 0.809949136609878
marginal = [[-245.3219901]]

# BA graph

## band pass

*theta = [0., 1., 4., 1., -6.]*

degree 1

unconstrained
beta1 = [0.02002668, 2.2202082 ]
noise = 0.28509182483425494
log-likelihood = [[-142.5878277]]

constrained
beta1 = [0.01565283, 2.24152686]
noise = 0.285091824834
constrained log-likelihood = [[-142.57379792]]

degree 2

beta2 = [-3.27699167e-05,  3.18928547e+00, -1.85618940e+00]
noise = 0.2370240904
log-likelihood = [[-96.944971]]

degree 3

beta3 = [ 0.17106661, -0.26708933, 10.14425437, -9.88533342]
noise = 0.210742085607
log-likelihood = [[-72.65905153]]

degree 4

beta4 = [ 4.64820715e-04,  1.95510593e+00,  2.34510875e-01,  6.07695500e+00, -8.19393751e+00]
noise = 0.25908400697
log-likelihood = [[-67.29521588]]

## low pass

degree 1

beta1 = [ 1.2291567,  -1.22718714]
noise std = 0.598849359066
constrained log-likelihood = [[-217.43845059]]

degree 2

beta2 = [ 1.46951222, -1.97320218,  1.00473307] 
noise = 0.3162569761516408
marginal = [[-217.62816777]] 

degree 3

beta3 = [ 1.3663117,  -1.41329927, -0.51916017,  0.95857868]
noise = 0.47634846644243367
marginal = [[-216.89314076]]

degree 4

beta4 = [ 1.26399243, -0.92871156, -1.86851298,  0.89769601,  0.80384401]
noise = 0.6303056341860226
log-likelihood = [[-216.433492]]