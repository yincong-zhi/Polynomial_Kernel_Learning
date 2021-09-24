import numpy as np
import matplotlib.pyplot as plt
from pygsp import graphs, filters, plotting, utils

G = graphs.Bunny()
#G.compute_fourier_basis()
#G.set_coordinates()

g = filters.Itersine(G, Nf=6)

fig, ax = plt.subplots(figsize=(10, 5))
g.plot(ax=ax)
_ = ax.set_title('Filter bank of mexican hat wavelets')
plt.show()

DELTA = 2
s = g.localize(DELTA)

fig = plt.figure(figsize=(20, 2.5))
for i in range(6):
    ax = fig.add_subplot(1, 6, i+1, projection='3d')
    G.plot_signal(s[:, i], ax=ax)
    _ = ax.set_title('Wavelet {}'.format(i+1))
    ax.set_axis_off()
#fig.tight_layout()
plt.show()

s = G.coords
s = g.filter(s)
s = np.linalg.norm(s, ord=2, axis=1)

fig = plt.figure(figsize=(10, 7))
for i in range(4):
    ax = fig.add_subplot(2, 2, i+1, projection='3d')
    G.plot_signal(s[:, i], ax=ax)
    title = 'Curvature estimation (scale {})'.format(i+1)
    _ = ax.set_title(title)
    ax.set_axis_off()
fig.tight_layout()
plt.show()


G = graphs.Ring(N=20)
G.estimate_lmax()
#G.set_coordinates('line1D')
#G.set_coordinates()
g = filters.MexicanHat(G)
s = g.localize(10)
fig, axes = plt.subplots(1, 2)
_ = g.plot(ax=axes[0])
_ = G.plot_signal(s[:,0], ax=axes[1])
plt.show()

G = graphs.ErdosRenyi(50,0.1)
G.compute_fourier_basis()

w,v = np.linalg.eig(G.L.todense())

def band(x):
    return x * np.exp(-x)

y = np.random.randn(50)
scales = np.arange(0.5, 2, 0.5)
ww = 0
for s in scales:
    ww += band(s*w)
z = v.dot(np.diag(band(ww))).dot(v.T).dot(y)

G.set_coordinates()
G.plot_signal(np.array(z).flatten())
plt.show()

x = np.arange(0,3,0.01)
y = 0
for s in scales:
    y += band(s*x)
    plt.plot(x, band(s*x))
y /= len(scales)
plt.plot(x, y, '--')
plt.title(r'$\alpha$')
plt.show()

def poly(beta, l):
    out = np.zeros(l.shape[0])
    for i, b in enumerate(beta):
        out += b*(l**i)
    return out

theta = np.array([0.,4.,-4.])
beta = np.array([-0.0586274,   1.51970768, -0.73679936])

l = np.arange(0,1,0.01)
plt.plot(l, poly(theta, l))
plt.plot(l, poly(theta, 0.5*l))

scales = [0.7838356,  1.98088708]
plt.plot(l, poly(beta, scales[1]*l)/np.max(poly(beta, scales[1]*l)), '.')
plt.plot(l, poly(beta, scales[0]*l)/np.max(poly(beta, scales[0]*l)), '.')
plt.legend(labels = ['ground truth 1', 'ground truth 2', 'recovered polynomial 1', 'recovered polynomial 2'])
plt.show()

scales = [1.75873903, 1.82720207]
plt.plot(l, scales[0]*l*np.exp(-scales[0]*l), '.')
plt.plot(l, scales[1]*l*np.exp(-scales[1]*l), '.')
plt.legend(labels = ['recovered exponential 1', 'recovered exponential 2'])

plt.show()
