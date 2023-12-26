import autograd.numpy as np
from autograd import grad
try:
    import matplotlib.pyplot as plt
except:
    pass
from pygsp import graphs
import argparse

from main import *

if parser.model in ('regularized_laplacian', 'diffusion', '1_random_walk', '3_random_walk', 'cosine'):
    G.compute_laplacian('normalized')
    G.compute_fourier_basis('normalized')
else:
    G.compute_laplacian()
    G.compute_fourier_basis()

def log_likelihood_base(alpha_log = 1., lengthscale = 10., variance = np.var(yn), noise = 0.05):
    alpha = np.log(1. + np.exp(alpha_log))
    lengthscale = np.log(1. + np.exp(lengthscale))
    variance = np.log(1. + np.exp(variance))
    noise = np.log(1. + np.exp(noise))
    # spatial
    I = np.eye(M)
    global B
    if parser.model == 'standard':
        B = I
    elif parser.model == 'laplacian':
        B = np.matmul(np.matmul(G.U, np.diag(np.concatenate((np.array([0]), np.sqrt(1./G.e[1:]))))), G.U.T)
    elif parser.model == 'local_averaging':
        B = np.matmul(np.linalg.inv(I + alpha*np.diag(G.d)), I + alpha*np.array(G.W.todense()))
    elif parser.model == 'global_filtering':
        B = np.linalg.inv((I + alpha*np.array(G.L.todense())))
    elif parser.model == 'regularized_laplacian':
        B = np.linalg.inv(I + alpha*np.array(G.L.todense()))
    elif parser.model == 'diffusion':
        B = np.matmul(G.U, np.matmul(np.diag(np.exp(-alpha*G.e)), G.U.T))
    elif parser.model == '1_random_walk': 
        B = (2+alpha)*I + np.array(G.L.todense())
    elif parser.model == '3_random_walk':
        B = np.matmul(G.U, np.matmul(np.diag(((2 + alpha) - G.e)**3), G.U.T))
    elif parser.model == 'cosine':
        B = np.matmul(G.U, np.matmul(np.diag(np.cos(np.pi*G.e/4.)), G.U.T))

    # temporal
    sqdist = np.sum(xn**2, 1).reshape(-1,1) + np.sum(xn**2,1) - 2*np.dot(xn,xn.T)
    temporal = np.exp(-0.5 * (1./lengthscale) * sqdist)
    K = variance*temporal
    # full covariance
    if parser.model in ('regularized_laplacian', 'diffusion', '1_random_walk', '3_random_walk', 'cosine'):
        CgN = np.kron(B, K) + (noise * np.eye(M*N))
    else:
        CgN = np.kron(np.matmul(B,B.T), K) + (noise * np.eye(M*N))
    L = np.linalg.cholesky(CgN + 1e-10*np.eye(N*M))
    alpha_L = np.linalg.solve(L,ttilde)

    log_likelihood = (- 0.5*np.linalg.slogdet(CgN)[1]) - (0.5 * np.matmul(alpha_L.T, alpha_L))
    return log_likelihood

def log_likelihood_posterior(x, mu, sigma):
    size = len(x)
    x_mu = np.matrix(x - mu)
    L = np.linalg.cholesky(sigma + 1e-10*np.eye(size))
    alpha_L = np.linalg.solve(L, x_mu)
    return -(float(size)/2.)*np.log(2.*np.pi) - (1./2.)*np.linalg.slogdet(sigma)[1] - (1./2.) * np.matmul(alpha_L.T, alpha_L)

if __name__ == '__main__':
    print('data: {}, training data: {}, baseline: {}'.format(data_name, N, parser.model))
    
    alpha = np.log(1.)
    lengthscale = np.mean(np.sum(np.square(xn), axis = 1))
    variance = np.var(yn)
    noise = 0.1 * lengthscale
    rate = 0.001 # alpha
    rate2 = 0.0001 # noise
    rate3 = 1. # lengthscale
    rate4 = 0.001 # variance

    print('log-likelihood =', log_likelihood_base(alpha, lengthscale, variance, noise))
    l_old = log_likelihood_base(alpha, lengthscale, variance, noise)
    dl = grad(log_likelihood_base, [0, 1, 2, 3])
    dld = dl(alpha, lengthscale, variance, noise)
    alpha += rate *np.array(dld[0])
    lengthscale += rate3 * np.array(dld[1])
    variance += rate4 * np.array(dld[2])
    noise += rate2 *np.array(dld[3])
    l = log_likelihood_base(alpha, lengthscale, variance, noise)
    print('log-likelihood =', l, 'alpha =', alpha, 'l =', lengthscale, 'v =', variance, 'n =', noise)
    while np.abs(l_old - l) > 0.001:
        l_old = l.copy()
        dld = dl(alpha, lengthscale, variance, noise)
        alpha += rate*np.array(dld[0])
        lengthscale += rate3 * np.array(dld[1])
        variance += rate4 * np.array(dld[2])
        noise += rate2 *np.array(dld[3])
        l = log_likelihood_base(alpha, lengthscale, variance, noise)
        print('log-likelihood =', l, 'alpha =', alpha, 'l =', lengthscale, 'v =', variance, 'n =', noise)

    print('log-likelihood =', log_likelihood_base(alpha, lengthscale, variance,noise))
    if parser.model not in ('standard', 'laplacian', 'cosine'):
        print('alpha =', alpha)
    print('l =', lengthscale, 'v =', variance, 'n =', noise)

    ll = []
    for i in range(1,num_test+1):
        xn1 = xn1T[((i-1)*size):(i*size),:]
        yn1 = yn1T[((i-1)*size):(i*size),:]

        MM = yn1.shape[0] # number of test signals

        alpha, lengthscale, variance, noise = np.log(1. + np.exp(alpha)), np.log(1.+np.exp(lengthscale)), np.log(1.+np.exp(variance)), np.log(1. + np.exp(noise))
        I = np.eye(M)
        if parser.model == 'standard':
            B = I
        elif parser.model == 'laplacian':
            B = np.matmul(np.matmul(G.U, np.diag(np.concatenate((np.array([0]), np.sqrt(1./G.e[1:]))))), G.U.T)
        elif parser.model == 'local_averaging':
            B = np.matmul(np.linalg.inv(I + alpha*np.diag(G.d)), I + alpha*np.array(G.W.todense()))
        elif parser.model == 'global_filtering':
            B = np.linalg.inv((I + alpha*np.array(G.L.todense())))
        elif parser.model == 'regularized_laplacian':
            B = np.linalg.inv(I + alpha*np.array(G.L.todense()))
        elif parser.model == 'diffusion':
            B = np.matmul(G.U, np.matmul(np.diag(np.exp(-alpha*G.e)), G.U.T))
        elif parser.model == '1_random_walk': 
            B = (2+alpha)*I + np.array(G.L.todense())
        elif parser.model == '3_random_walk':
            B = np.matmul(G.U, np.matmul(np.diag(((2 + alpha) - G.e)**3), G.U.T))
        elif parser.model == 'cosine':
            B = np.matmul(G.U, np.matmul(np.diag(np.cos(np.pi*G.e/4.)), G.U.T))

        sqdist = np.sum(xn**2, 1).reshape(-1,1) + np.sum(xn**2,1) - 2*np.dot(xn,xn.T)
        temporal = np.exp(-0.5 * (1./lengthscale) * sqdist)
        K = variance * temporal

        sqdist = np.sum(xn**2, 1).reshape(-1,1) + np.sum(xn1**2,1) - 2*np.dot(xn,xn1.T)
        temporal = np.exp(-0.5 * (1./lengthscale) * sqdist)
        k = variance * temporal

        sqdist = np.sum(xn1**2, 1).reshape(-1,1) + np.sum(xn1**2,1) - 2*np.dot(xn1,xn1.T)
        temporal = np.exp(-0.5 * (1./lengthscale) * sqdist)
        k_star = variance * temporal
    
        if parser.model in ('regularized_laplacian', 'diffusion', '1_random_walk', '3_random_walk', 'cosine'):
            CgN = np.kron(B, K) + (noise * np.eye(M*N))
            D = np.kron(B, k)
            F = np.kron(B, k_star) + (noise * np.eye(M*MM))
        else:
            CgN = np.kron(np.matmul(B,B.T), K) + (noise * np.eye(M*N))
            D = np.kron(np.matmul(B,B.T), k)
            F = np.kron(np.matmul(B,B.T), k_star) + (noise * np.eye(M*MM))

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