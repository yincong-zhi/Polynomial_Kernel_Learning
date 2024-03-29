import autograd.numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass
from pygsp import graphs
from autograd import grad
from invwishart import invwishartrand

# graph dimension
M = 30
# number of signals
N = 15

G = graphs.Sensor(M, seed = 3)
#G = graphs.ErdosRenyi(M, 0.05, seed = 1)

np.random.seed(3)

def poly(beta, l):
    out = np.zeros(l.shape[0])
    for i, b in enumerate(beta):
        out += b*(l**i)
    return out

#G.compute_laplacian('normalized')
G.compute_laplacian()
w,_ = np.linalg.eig(G.L.todense())
GL = 1./np.max(w) * np.array(G.L.todense())
w,v = np.linalg.eig(GL)
l = np.arange(0.,np.max(w)+0.01,0.01)

# data spectrum shape (change for different types of data)
#theta = np.array([0., 1., 4., 1., -6.])
theta = np.array([1., -1.5, (1.5**2)/2., -(1.5**3)/6., (1.5**4)/24.])

A = GL
AA = np.zeros((M,M))
for i, j in enumerate(theta):
    AA += j * np.linalg.matrix_power(A,i)

C_s = invwishartrand(N, np.eye(N))
# sample M times from multivariate normal
y = np.random.multivariate_normal(np.zeros(len(C_s)), C_s, size=M)
# define graph filter
z = AA.dot(y)

# add noise to form data
noise = 0.1*np.random.randn(M,N)
yn = z.T + noise.T
ttilde = yn.flatten(order = 'F').reshape(-1,1)
# temporal kernel from wishart
K = C_s

gamma = 0.
def log_likelihood_fun(beta):
    # spatial
    B = np.zeros((G.N,G.N))
    for i in range(len(beta)):
        B += beta[i] * np.linalg.matrix_power(GL, i)
    # full covariance
    CgN = np.kron(np.matmul(B, B.T), K) + (1/100. * np.eye(M*N))
    L = np.linalg.cholesky(CgN + 1e-10*np.eye(N*M))
    alpha_L = np.linalg.solve(L,ttilde)

    log_likelihood = (- 0.5*np.linalg.slogdet(CgN)[1]) - (0.5 * np.matmul(alpha_L.T, alpha_L))
    return log_likelihood

from itertools import product

def initialize_beta(degree, grid = np.arange(-6.,7.,2.)):
    # find best initial values of beta's using a grid search
    l_old = np.array([[-100000]])
    for b in product(grid, repeat = degree):
        b = np.array(b)
        l = log_likelihood_fun(b)
        if l >= l_old:
            beta = b
            l_old = l.copy()
    #print 'initial beta =', beta
    print('initial log-likelihood =', log_likelihood_fun(beta))
    return beta

def unconstrained_search(degree, beta = None, rate = 0.0001, tolerance = 0.0001):
    if beta is None:
        beta = initialize_beta(degree)

    # find optimal using gradient ascent
    l_old = log_likelihood_fun(beta)
    dl = grad(log_likelihood_fun, [0])
    dld = dl(beta)
    beta += rate *np.array(dld[0])
    l = log_likelihood_fun(beta)
    print('log likelihood =', l, 'beta =', beta)
    while np.abs(l_old - l) > tolerance:
        l_old = l.copy()
        dld = dl(beta)
        beta += rate*np.array(dld[0])
        l = log_likelihood_fun(beta)
        print('log-likelihood =', l, 'beta =', beta)
    print('unconstrained beta =', beta)
    return beta

# objective: - log-likelihood + lagrange multiplier term
gamma = 0.
def dual(beta, log_lagrange):
    # - log-likelihood + lagrangian
    B = np.zeros((G.N,G.N))
    for i in range(len(beta)):
        B += beta[i] * np.linalg.matrix_power(GL, i)
    # full covariance
    CgN = np.kron(np.matmul(B, B.T), K) + (1/10. * np.eye(M*N))
    L = np.linalg.cholesky(CgN + 1e-10*np.eye(N*M))
    alpha_L = np.linalg.solve(L,ttilde)

    # construct vandermonde matrix
    B_eig = np.zeros((w.shape[0], beta.shape[0]))
    for i in range(len(beta)):
        B_eig[:,i] = w.flatten()**i
    lagrange = np.exp(log_lagrange)
    lagrangian = - np.matmul(np.matmul(lagrange.T, B_eig), beta.reshape(-1,1))

    log_likelihood = (- 0.5*np.linalg.slogdet(CgN)[1]) - (0.5 * np.matmul(alpha_L.T, alpha_L)) - gamma*np.sum(np.square(beta))
    return - log_likelihood + lagrangian

def constrained_search(beta, lagrange, rate = 0.0001, rate2 = 0.1, tolerance = 0.0001):
    '''
    used with unconstrained search
    '''
    l_lagrange_old = np.array([-1000000.])
    l_lagrange = dual(beta, lagrange)
    while np.abs(l_lagrange_old - l_lagrange) > 0.001:
        # update beta until convergence
        l_beta_old = dual(beta, lagrange)
        dl = grad(dual, [0])
        dld = dl(beta, lagrange)
        beta -= rate * dld[0]
        l_beta = dual(beta, lagrange)
        print('updating beta')
        while np.abs(l_beta_old - l_beta) > tolerance:
            l_beta_old = l_beta.copy()
            dld = dl(beta, lagrange)
            beta -= rate * dld[0]
            l_beta = dual(beta, lagrange)
            #print 'beta =', beta, l_beta
        print('beta =', beta, l_beta)
        # update lagrange once
        l_lagrange_old = l_lagrange.copy()
        dl = grad(dual, [1])
        dld = dl(beta, lagrange)
        lagrange += rate2 * dld[0]
        l_lagrange = dual(beta, lagrange)
        print('updating lagrange,', l_lagrange)
    return beta

if __name__ == '__main__':
    beta_unconstrained = []
    beta_constrained = []
    #print('here')
    for degree in range(2,6):
        beta_initial = initialize_beta(degree, grid = np.arange(-3.,4.,1.))
        beta = unconstrained_search(degree, beta = beta_initial.copy())
        beta_unconstrained.append(beta.copy())
        lagrange = 0.1*np.ones((len(w),1))
        beta = constrained_search(beta, lagrange, rate = 0.0001, tolerance = 0.0001)
        beta_constrained.append(beta.copy())
    
    print(beta_unconstrained)
    print(beta_constrained)
