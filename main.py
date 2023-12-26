import autograd.numpy as np
import scipy as sp
import scipy.io
from autograd import grad
try:
    import matplotlib.pyplot as plt
except:
    pass
from pygsp import graphs
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="fmri", type=str, help='fmri, uber, weather')
parser.add_argument("--training", default=42, type=int, help='fmri: 21 or 42, uber: 10 or 20, weather: 15 or 30')
parser.add_argument("--degree", default=3, type=int, help='polynomial degree')
parser.add_argument("--constrained", default='on', type=str, help='change to False to compute posterior directly from unconstrained solution')
parser.add_argument("--model", default="standard", type=str, help='baseline model (only needed for baselines.py): standard, laplacian, local_averaging, global_filtering, regularized_laplacian, diffusion, 1_random_walk, 3_random_walk, cosine')

parser = parser.parse_args()
data_name = parser.data
N = parser.training
from import_data import *

def poly(beta, l):
    out = np.zeros(l.shape[0])
    for i, b in enumerate(beta):
        out += b*(l**i)
    return out

def log_likelihood_fun(beta, lengthscale = 10., noise = 0.05, grid = False):
    # do grid search on the smaller training size if larger is selected
    if grid:
        if data_name == 'uber':
            half_d = 10
        elif data_name == 'fmri':
            half_d = 21
        elif data_name == 'weather':
            half_d = 15
        xn_in, yn_in = xn[:half_d,:], yn[:half_d,:]
        N_in, ttilde = xn_in.shape[0], yn_in.reshape(-1,1, order = 'F')
    else:
        xn_in, yn_in, N_in, ttilde = xn, yn, N, yn.reshape(-1,1, order = 'F')
    # spatial
    B = np.sum(np.array([beta[i] * np.linalg.matrix_power(GL, i) for i in range(len(beta))]), axis=0)
    # temporal
    lengthscale = np.log(1.+np.exp(lengthscale))
    noise = np.log(1.+np.exp(noise))
    sqdist = np.sum(xn_in**2, 1).reshape(-1,1) + np.sum(xn_in**2,1) - 2*np.dot(xn_in,xn_in.T)
    temporal = np.exp(-0.5 * (1./lengthscale) * sqdist)
    K = np.var(yn_in)*temporal
    # full covariance
    CgN = np.kron(np.matmul(B, B.T), K) + (noise * np.eye(M*N_in))
    L = np.linalg.cholesky(CgN + 1e-10*np.eye(N_in*M))
    alpha_L = np.linalg.solve(L,ttilde)

    log_likelihood = (- 0.5*np.linalg.slogdet(CgN)[1]) - (0.5 * np.matmul(alpha_L.T, alpha_L))
    return log_likelihood

from itertools import product

def initialize_beta(degree, grid = np.arange(-6.,7.,1.), lengthscale = 10., noise = np.sqrt(10.)):
    if (data_name == 'uber' and N > 10) or (data_name == 'fmri' and N > 21) or (data_name == 'weather' and N > 15):
        half = True
    else:
        half = False
    # find best initial values of beta's using a grid search
    l_old = np.array([[-100000]])
    for b in product(grid, repeat = degree):
        b = np.array(b)
        l = log_likelihood_fun(b, lengthscale = lengthscale, noise = noise, grid = half)
        if l >= l_old:
            beta = b
            l_old = l.copy()
    #print('initial beta =', beta)
    print('initial log-likelihood =', log_likelihood_fun(beta, lengthscale=lengthscale, noise=noise))
    return beta

def unconstrained_search(degree, beta = None, lengthscale = 10., noise = np.sqrt(10.), rate = 0.0001, rate2 = 0.0001, rate3 = 1., tolerance = 0.0001):
    if beta is None:
        beta = initialize_beta(degree)

    # find optimal using gradient ascent
    l_old = log_likelihood_fun(beta, lengthscale, noise)
    dl = grad(log_likelihood_fun, [0, 1, 2])
    dld = dl(beta, lengthscale, noise)
    beta += rate *np.array(dld[0])
    lengthscale += rate3 * np.array(dld[1])
    noise += rate2 *np.array(dld[2])
    l = log_likelihood_fun(beta, lengthscale, noise)
    print('log-likelihood =', l, 'beta =', beta, 'l =', lengthscale, 'n =', noise)
    while np.abs(l_old - l) > tolerance:
        l_old = l.copy()
        dld = dl(beta, lengthscale, noise)
        beta += rate*np.array(dld[0])
        lengthscale += rate3 * np.array(dld[1])
        noise += rate2 *np.array(dld[2])
        l = log_likelihood_fun(beta, lengthscale, noise)
        print('log-likelihood =', l, 'beta =', beta, 'l =', lengthscale, 'n =', noise)
    print('unconstrained beta =', beta)
    return beta, lengthscale, noise

def dual(beta, log_lagrange, lengthscale = 1000., noise = np.sqrt(100.)):
    # - log-likelihood + lagrangian
    B = np.sum(np.array([beta[i] * np.linalg.matrix_power(GL, i) for i in range(len(beta))]), axis=0)
    # temporal
    lengthscale = np.log(1.+np.exp(lengthscale))
    noise = np.log(1.+np.exp(noise))
    sqdist = np.sum(xn**2, 1).reshape(-1,1) + np.sum(xn**2,1) - 2*np.dot(xn,xn.T)
    temporal = np.exp(-0.5 * (1./lengthscale) * sqdist)
    K = np.var(yn)*temporal
    # full covariance
    CgN = np.kron(np.matmul(B, B.T), K) + (noise * np.eye(M*N))
    L = np.linalg.cholesky(CgN + 1e-10*np.eye(N*M))
    alpha_L = np.linalg.solve(L,ttilde)
    log_likelihood = (- 0.5*np.linalg.slogdet(CgN)[1]) - (0.5 * np.matmul(alpha_L.T, alpha_L))
    
    # construct vandermonde matrix
    B_eig = np.zeros((w.shape[0], beta.shape[0]))
    for i in range(len(beta)):
        B_eig[:,i] = w.flatten()**i
    lagrange = np.log(1. + np.exp(log_lagrange))
    lagrangian = - np.matmul(np.matmul(lagrange.T, B_eig), beta.reshape(-1,1))

    return - log_likelihood + lagrangian

def constrained_search(beta, lagrange, rate = 0.0001, rate2 = 0.1, tolerance = 0.001, lengthscale = 1000., noise = np.sqrt(100.)):
    l_lagrange_old = np.array([-1000000.])
    l_lagrange = dual(beta, lagrange, lengthscale, noise)
    while np.abs(l_lagrange_old - l_lagrange) > 0.001:
        # update beta until convergence
        l_beta_old = dual(beta, lagrange, lengthscale, noise)
        dl = grad(dual, [0])
        dld = dl(beta, lagrange, lengthscale, noise)
        beta -= rate * dld[0]
        l_beta = dual(beta, lagrange, lengthscale, noise)
        print('updating beta')
        while np.abs(l_beta_old - l_beta) > tolerance:
            l_beta_old = l_beta.copy()
            dld = dl(beta, lagrange, lengthscale, noise)
            beta -= rate * dld[0]
            l_beta = dual(beta, lagrange, lengthscale, noise)
            print('beta =', beta, l_beta, np.abs(l_beta_old - l_beta))
        #print 'beta =', beta, l_beta
        # update lagrange once
        l_lagrange_old = l_lagrange.copy()
        dl = grad(dual, [1])
        dld = dl(beta, lagrange, lengthscale, noise)
        lagrange += rate2 * dld[0]
        l_lagrange = dual(beta, lagrange, lengthscale, noise)
        print('updating lagrange,', l_lagrange)
        print('min, max =', np.min(lagrange), np.max(lagrange))
    return beta

def log_likelihood_posterior(x, mu, sigma):
    size = len(x)
    x_mu = np.matrix(x - mu)
    L = np.linalg.cholesky(sigma + 1e-10*np.eye(size))
    alpha_L = np.linalg.solve(L, x_mu)
    return -(float(size)/2.)*np.log(2.*np.pi) - (1./2.)*np.linalg.slogdet(sigma)[1] - (1./2.) * np.matmul(alpha_L.T, alpha_L)
    
if __name__ == '__main__':
    degree = parser.degree + 1
    print('data: {}, degree: {}, training data: {}, constrained: {}'.format(data_name, degree-1, N, parser.constrained))
    
    print('initializing beta')
    l_mean = np.mean(np.sum(np.square(xn), axis = 0))
    beta_initial = initialize_beta(degree, grid = np.arange(-6,7,1.), lengthscale = l_mean, noise = l_mean/10.)
    #beta_initial = np.array([ 1., -1., -4.,  0.,  5.])
    print(beta_initial)

    beta, lengthscale, noise = unconstrained_search(degree, beta_initial.copy(), lengthscale = l_mean, noise = l_mean/10., rate = 0.0001, rate2 = 0.0001, rate3 = 1., tolerance = 0.001)

    print('unconstrained beta =', beta)
    print('l = {}, n = {}'.format(lengthscale, noise))
    
    if parser.constrained == 'on':
        print('constrained optimization')
        beta += 1.
        lagrange = -1.*np.ones((len(w),1))
        beta = constrained_search(beta, lagrange, lengthscale = lengthscale, noise = noise, rate = 0.0001, rate2 = 1., tolerance = 0.001)
        
        print('beta = ', beta)
        print('l = {}, n = {}'.format(lengthscale, noise))
        print('marginal = ', log_likelihood_fun(beta, lengthscale, noise))
    
    '''
    compute posteriors
    '''
    ll = []
    lengthscale, noise = np.log(1.+np.exp(lengthscale)), np.log(1.+np.exp(noise))
    for i in range(1,num_test+1):
        xn1 = xn1T[((i-1)*size):(i*size),:]
        yn1 = yn1T[((i-1)*size):(i*size),:]

        MM = yn1.shape[0] # number of test signals
        
        B = np.sum(np.array([beta[j] * np.linalg.matrix_power(GL, j) for j in range(len(beta))]), axis=0)

        variance = np.var(yn)
        sqdist = np.sum(xn**2, 1).reshape(-1,1) + np.sum(xn**2,1) - 2*np.dot(xn,xn.T)
        temporal = np.exp(-0.5 * (1./lengthscale) * sqdist)
        K = variance * temporal

        sqdist = np.sum(xn**2, 1).reshape(-1,1) + np.sum(xn1**2,1) - 2*np.dot(xn,xn1.T)
        temporal = np.exp(-0.5 * (1./lengthscale) * sqdist)
        k = variance * temporal

        sqdist = np.sum(xn1**2, 1).reshape(-1,1) + np.sum(xn1**2,1) - 2*np.dot(xn1,xn1.T)
        temporal = np.exp(-0.5 * (1./lengthscale) * sqdist)
        k_star = variance * temporal
        
        CgN = np.kron(np.matmul(B, B.T), K) + (noise * np.eye(M*N))
        D = np.kron(np.matmul(B, B.T), k)
        F = np.kron(np.matmul(B, B.T), k_star) + (noise * np.eye(M*MM))

        # cholesky decomposition
        L = np.linalg.cholesky(CgN + 1e-10*np.eye(N*M))

        # posterior mean and sd
        LT = np.linalg.solve(L, ttilde)
        muN1 = D.T.dot(np.linalg.solve(L.T, LT))
        LT = np.linalg.solve(L, D)
        sigmaN1 = F - D.T.dot(np.linalg.solve(L.T,LT))

        log_likelihood = log_likelihood_posterior(yn1.reshape(-1,1, order = 'F'), muN1, sigmaN1)

        ll.append(float(log_likelihood/size))
    print(ll)
    print('mean = {}, se = {}'.format(np.mean(ll), np.std(ll)/np.sqrt(len(ll))))