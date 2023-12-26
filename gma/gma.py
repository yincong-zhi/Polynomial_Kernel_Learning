import autograd.numpy as np
try:
    import matplotlib.pyplot as plt
except:
    pass
from pygsp import graphs
from autograd import grad
#from invwishart import invwishartrand

# number of signals
N = 20
SNR = 10.

# graph dimension
M = 30
#G = graphs.Sensor(M, seed = 1)
G = graphs.ErdosRenyi(M, 0.2, seed = 1)
#G = graphs.ErdosRenyi(M, 0.05, seed = 1)
#G = graphs.BarabasiAlbert(M, 10, 5, seed = 1)
np.random.seed(1)

def poly(beta, l):
    out = np.zeros(l.shape[0])
    for i, b in enumerate(beta):
        out += b*(l**i)
    return out

G.compute_laplacian()
w,_ = np.linalg.eig(G.L.todense())
GL = 1./np.max(w) * np.array(G.L.todense())
w,v = np.linalg.eig(GL)
l = np.arange(0.,np.max(w)+0.01,0.01)

# data spectrum shape (change for different types of data)
theta = np.array([0., 1., 4., 1., -6.])
#theta = np.array([1., -1.5, (1.5**2)/2., -(1.5**3)/6., (1.5**4)/24.])

# compute and apply ground truth filter
y = np.random.normal(scale=1., size = (M,N))
A = GL
AA = np.zeros((M,M))
for i, j in enumerate(theta):
    AA += j * np.linalg.matrix_power(A,i)
z = AA.dot(y)

# standardize
z /= np.std(z)

signal_var = np.var(z)

# add noise to form data
noise = np.sqrt(signal_var * 10**(-SNR/10.) )*np.random.randn(M,N)
yn = z.T + noise.T
ttilde = yn.flatten(order = 'F').reshape(-1,1)
# use identity (white noise) as temporal kernel
K = np.var(yn)*np.eye(N)

plt.plot(w, v.T.dot(yn.T), '.')
plt.show()

gamma = 0.
def log_likelihood_fun(beta, noise = np.sqrt(0.1)):
    # spatial
    B = np.zeros((G.N,G.N))
    for i in range(len(beta)):
        B += beta[i] * np.linalg.matrix_power(GL, i)
    # full covariance
    CgN = np.kron(np.matmul(B, B.T), K) + (np.square(noise) * np.eye(M*N))
    L = np.linalg.cholesky(CgN + 1e-10*np.eye(N*M))
    alpha_L = np.linalg.solve(L,ttilde)

    log_likelihood = (- 0.5*np.linalg.slogdet(CgN)[1]) - (0.5 * np.matmul(alpha_L.T, alpha_L))# - gamma*np.sum(np.square(beta))
    if log_likelihood == np.inf:
        return - np.inf
    else:
        return log_likelihood

from itertools import product

def initialize_beta(degree, grid = np.arange(-6.,7.,2.), noise = np.sqrt(0.1)):
    # find best initial values of beta's using a grid search
    l_old = np.array([[-100000]])
    for b in product(grid, repeat = degree):
        b = np.array(b)
        l = log_likelihood_fun(b, noise = noise)
        if l >= l_old:
            beta = b
            l_old = l.copy()
    #print 'initial beta =', beta
    print('initial log-likelihood =', log_likelihood_fun(beta, noise = noise))
    return beta

def unconstrained_search(degree, beta = None, noise = np.sqrt(0.1), rate = 0.0001, rate2 = 0.01, tolerance = 0.0001):
    if beta is None:
        beta = initialize_beta(degree)

    # find optimal using gradient ascent
    l_old = log_likelihood_fun(beta, noise)
    dl = grad(log_likelihood_fun, [0, 1])
    dld = dl(beta, noise)
    beta += rate *np.array(dld[0])
    noise += rate2 *np.array(dld[1])
    l = log_likelihood_fun(beta, noise)
    print('log-likelihood =', l, 'beta =', beta, 'noise =', noise)
    while np.abs(l_old - l) > tolerance:
        l_old = l.copy()
        dld = dl(beta, noise)
        beta += rate*np.array(dld[0])
        noise += rate2 *np.array(dld[1])
        l = log_likelihood_fun(beta, noise)
        print('log-likelihood =', l, 'beta =', beta, 'noise =', noise)
    print('unconstrained beta =', beta)
    return beta, noise

# objective: - log-likelihood + lagrange multiplier term
def dual(beta, log_lagrange, noise = np.sqrt(0.1)):
    # - log-likelihood + lagrangian
    B = np.zeros((G.N,G.N))
    for i in range(len(beta)):
        B += beta[i] * np.linalg.matrix_power(GL, i)
    # full covariance
    CgN = np.kron(np.matmul(B, B.T), K) + (np.square(noise) * np.eye(M*N))
    L = np.linalg.cholesky(CgN + 1e-10*np.eye(N*M))
    alpha_L = np.linalg.solve(L,ttilde)

    # construct vandermonde matrix
    B_eig = np.zeros((w.shape[0], beta.shape[0]))
    for i in range(len(beta)):
        B_eig[:,i] = w.flatten()**i
    lagrange = np.exp(log_lagrange)
    lagrangian = - np.matmul(np.matmul(lagrange.T, B_eig), beta.reshape(-1,1))

    log_likelihood = (- 0.5*np.linalg.slogdet(CgN)[1]) - (0.5 * np.matmul(alpha_L.T, alpha_L))
    return - log_likelihood + lagrangian

def constrained_search(beta, lagrange, noise = np.sqrt(0.1), rate = 0.0001, rate2 = 0.1, tolerance = 0.0001):

    l_lagrange_old = np.array([-1000000.])
    l_lagrange = dual(beta, lagrange, noise)
    while np.abs(l_lagrange_old - l_lagrange) > 0.001:
        # update beta until convergence
        l_beta_old = dual(beta, lagrange, noise)
        dl = grad(dual, [0, 2])
        dld = dl(beta, lagrange, noise)
        beta -= rate * dld[0]
        #noise -= 10. * rate * dld[1]
        l_beta = dual(beta, lagrange, noise)
        print('updating beta')
        while np.abs(l_beta_old - l_beta) > tolerance:
            l_beta_old = l_beta.copy()
            dld = dl(beta, lagrange, noise)
            beta -= rate * dld[0]
            #noise -= 10. * rate * dld[1]
            l_beta = dual(beta, lagrange, noise)
            print(l_beta, 'beta =', beta, 'difference =', np.abs(l_beta_old - l_beta))
        #print l_beta, 'beta =', beta
        # update lagrange once
        l_lagrange_old = l_lagrange.copy()
        dl = grad(dual, [1])
        dld = dl(beta, lagrange, noise)
        lagrange += rate2 * dld[0]
        l_lagrange = dual(beta, lagrange, noise)
        print('updating lagrange,', l_lagrange)
    return beta, noise

if __name__ == '__main__':
    degree = 3 # initialize degree 4 or more will take quite a while

    noise = np.sqrt(0.1) # noise std, change accordingly
    beta_initial = initialize_beta(degree, grid = np.arange(-10,11,1.), noise = noise)
    
    beta, noise = unconstrained_search(degree, beta = beta_initial.copy(), noise = noise, rate = 0.0001, rate2 = 0.001, tolerance  = 0.0001)
    
    l = np.arange(0,1.01,0.01)
    if any(poly(beta, l) < 0):
        print('solution not psd, needs constrained search')
        beta += 1.
        lagrange = 0.1*np.ones((len(w),1))
        beta, noise = constrained_search(beta, lagrange, noise, rate = 0.0001, rate2 = 0.1, tolerance = 0.0001)

    print('beta = {}, noise = {}'.format(beta, noise))
    print('log-likelihood =', log_likelihood_fun(beta, noise))
