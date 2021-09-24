def poly(beta, l):
    out = np.zeros(l.shape[0])
    for i, b in enumerate(beta):
        out += b*(l**i)
    return out

font = {'size'   : 15}
matplotlib.rc('font', **font)

plt.plot(w, v.T.dot(yn.T), '.')
plt.title('weather data gft coefficients')
plt.xlabel(r'eigenvalues $\lambda$'), plt.ylabel(r'$U^\top y$')
plt.show()

l = np.arange(0,1.01,0.01)
plt.plot(l, poly(beta1,l)/np.max(poly(beta1,l)))
plt.plot(l, poly(beta2,l)/np.max(poly(beta2,l)))
plt.plot(l, poly(beta3,l)/np.max(poly(beta3,l)))
plt.plot(l, poly(beta4,l)/np.max(poly(beta4,l)))
plt.plot(l, poly(theta,l)/np.max(poly(theta,l)), '--')
plt.grid()
plt.legend(labels = [r'$g_1(\lambda)$, l = 645.31', r'$g_2(\lambda)$, l = 695.17', r'$g_3(\lambda)$, l = 695.42', r'$g_4(\lambda)$, l = 695.04', r'$\theta(\lambda)$'])
plt.xlabel(r'eigenvalues $\lambda$'), plt.ylabel(r'$g(\lambda)$')
plt.title(r'scaled polynomials (marginal log-likelihood l)')
plt.show()

log_likelihoods = [-307.08966448, -272.16061707, -201.82558506, -199.06508327]
plt.plot(range(1,5), log_likelihoods)
plt.plot(range(1,5), log_likelihoods, 'x')
plt.grid()
plt.xticks(range(1, 5))
plt.xlabel(r'polynomial degree')#, plt.ylabel('log-likelihoods')
plt.title('polynomial kernels gp marginal log-likelihoods')
plt.show()


l = np.arange(0,1.01,0.01)
plt.plot(l, poly(beta5,l)/np.max(poly(beta5,l)))
plt.plot(l, poly(beta10,l)/np.max(poly(beta10,l)))
plt.plot(l, poly(beta15,l)/np.max(poly(beta15,l)))
#plt.plot(l, poly(beta20,l)/np.max(poly(beta20,l)))
#plt.plot(l, poly(beta25,l)/np.max(poly(beta25,l)))
plt.plot(l, poly(theta,l)/np.max(poly(theta,l)), '--')
plt.grid()
plt.legend(labels = [r'5 signals', r'10 signals', r'15 signals', r'$\theta(\lambda)$'])
plt.xlabel(r'eigenvalues $\lambda$'), plt.ylabel(r'$g(\lambda)$')
plt.title(r'SNR 2.5')
plt.show()

log_likelihoods = [-28.39863634, -30.17847888, -30.998405, -30.53660218]
plt.plot(range(5,25,5), log_likelihoods)
plt.plot(range(5,25,5), log_likelihoods, 'x')
plt.grid()
plt.xticks(range(5,25,5))
plt.xlabel(r'training sample size'), plt.ylabel('log-likelihoods')
plt.title('average marginal log-likelihoods')
plt.show()


l = np.array([[-5.66325013, -12.27019656, -22.71787782, -43.23564822, -62.10290363],
              [-1.81542481,  -6.63579569, -16.36371957, -34.38940414, -54.70060841],
              [-2.05431458,  -6.58717027, -14.70008824, -30.74965657, -51.87374752],
              [-0.80676775,  -5.58213452, -15.39968855, -33.70900814, -55.89452529],
              [-3.00841267,  -7.84527132, -17.83651223, -35.64595179, -57.50587682],
              [-3.74616107,  -8.51427783, -18.17187535, -36.13365022, -58.41797411]])

array([[ -5.66325013,  -1.81542481,  -2.05431458,  -0.80676775,  -3.00841267,  -3.74616107],
       [-12.27019656,  -6.63579569,  -6.58717027,  -5.58213452,  -7.84527132,  -8.51427783],
       [-22.71787782, -16.36371957, -14.70008824, -15.39968855,  -17.83651223, -18.17187535],
       [-43.23564822, -34.38940414, -30.74965657, -33.70900814,  -35.64595179, -36.13365022],
       [-62.10290363, -54.70060841, -51.87374752, -55.89452529,  -57.50587682, -58.41797411]])

nn = np.array([2**i for i in range(0,6)])
ss = np.array([5*i for i in range(-2,3)])

# 3d plot
from mpl_toolkits.mplot3d import Axes3D
x, y = np.meshgrid(ss, np.log(nn))
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x,y,l, color = 'red')
plt.tight_layout()
ax.set_xticks(ss)
#ax.set_yticks(np.arange(6))
labels = [r'$2^{}$'.format(i) for i in range(6)]
ax.set_yticklabels(labels)
ax.set_xlabel('SNR (db)')
ax.set_ylabel('training size')
ax.set_zlabel('log-likelihoods')
ax.set_title('polynomial kernel GP marginal log-likelihoods')
plt.show()

# imshow
fig, ax = plt.subplots(1,1)
img = ax.imshow(l.T)
ax.set_xticks([0,1,2,3,4,5])
ax.set_yticks([0,1,2,3,4])
ax.set_yticklabels(ss)
ax.set_xticklabels([r'$2^{}$'.format(n) for n in range(6)])
fig.colorbar(img)
plt.tight_layout()
plt.ylabel('SNR (db)')
plt.xlabel('training size')
plt.title('marginal log-likelihoods')
plt.show()

x_y = np.zeros(M)
x_y[:g_split] = 1
G.plot_signal(x_y)

l = np.arange(0,1.01,0.01)
beta2 = [0.52600555, 0.39954975]
beta3 = [ 0.81406568, -2.49094452,  3.19831824]
beta4 = [ 0.98061968, -5.32475351,  1.10448791,  5.2128654 ]
plt.plot(l, poly(beta2,l)/np.max(poly(beta2,l)))
plt.plot(l, poly(beta3,l)/np.max(poly(beta3,l)))
plt.plot(l, poly(beta4,l)/np.max(poly(beta4,l)))
plt.grid()
plt.show()

plt.plot(w, (v.T.dot(yn.T)), '.')
#plt.plot(w, np.abs(v.T.dot(yn.T)), '.')



l = np.arange(0,1.01,0.01)
plt.plot(l, poly(beta20,l)/np.max(poly(beta20,l)))
plt.plot(l, poly(beta15,l)/np.max(poly(beta15,l)))
plt.plot(l, poly(beta10,l)/np.max(poly(beta10,l)))
plt.plot(l, poly(beta5,l)/np.max(poly(beta5,l)))
plt.plot(l, poly(beta0,l)/np.max(poly(beta0,l)))
plt.plot(l, poly(theta,l)/np.max(poly(theta,l)), '--')
plt.grid()
plt.legend(labels = [r'SNR 20, l = -42.72', r'SNR 15, l = -70.99', r'SNR 10, l = -106.95', r'SNR 5, l = -163.60', r'SNR 0, l = -248.40', r'$\theta(\lambda)$'], loc = 'best')
plt.xlabel(r'eigenvalues $\lambda$'), plt.ylabel(r'$g(\lambda)$')
plt.title(r'scaled polynomials')
plt.show()


l = np.arange(0,1.01,0.01)
plt.plot(l, poly(beta1,l)/np.max(poly(beta1,l)))
plt.plot(l, poly(beta2,l)/np.max(poly(beta2,l)))
plt.plot(l, (poly(beta3,l)+ 0.7646615927500501)/(np.max(poly(beta3,l)) + 0.7646615927500501))
plt.grid()
plt.legend(labels = [r'degree 1', r'degree 2', r'degree 3'], loc = 'best')
plt.xlabel(r'eigenvalues $\lambda$'), plt.ylabel(r'$g(\lambda)$')
plt.title(r'Traffic')
plt.show()

