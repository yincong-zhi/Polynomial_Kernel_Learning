import autograd.numpy as np
import scipy as sp
import scipy.io
from autograd import grad
try:
    import matplotlib.pyplot as plt
except:
    pass
from pygsp import graphs
import networkx as nx
import argparse

from main import data_name, N

if data_name == 'uber':
    uber = scipy.io.loadmat('data/uber_data.mat')
    coord = np.concatenate((uber['lon'], uber['lat']), axis = 1)

    data = uber['X1']
    data = (data - np.mean(data))/np.std(data)

    GG = graphs.NNGraph(coord, k = 4)
    GG.set_coordinates(coord)

    # graph the output is on
    np.random.seed(8) #8, 24, 36
    out_nodes = np.random.choice(np.arange(0,29),20, replace=False)
    in_nodes = np.delete(np.arange(0,29), out_nodes)
    G = graphs.Graph(GG.W.todense()[out_nodes[:,None],out_nodes])
    G.set_coordinates(coord[out_nodes])

    # train-test split
    train_id = np.random.randint(0,744,20)
    test_id = np.delete(np.arange(0,744), train_id)
    # only use first 10 if N = 10
    train_id = train_id[:N]

    xn = data[in_nodes[:,None],train_id].T
    yn = data[out_nodes[:,None],train_id].T
    ttilde = yn.reshape(-1,1, order = 'F')

    xn1T = data[in_nodes[:,None],test_id].T
    yn1T = data[out_nodes[:,None],test_id].T

    M = G.N
    size = 10
    num_test = 20

elif data_name == 'fmri':
    A = scipy.io.loadmat('data/A_cerebellum.mat')['A']
    signal = scipy.io.loadmat('data/signal_set_cerebellum.mat')['F2']

    #N = 42

    signal = (signal - np.mean(signal))/np.std(signal)

    np.random.seed(59) #59
    nodes_id = np.random.randint(0,4465,500)
    W = A[nodes_id[:,None], nodes_id].todense()
    G = nx.from_numpy_matrix(W)
    sizes = [len(i) for i in nx.connected_components(G)]
    idx = sizes.index(max(sizes))
    compon = [i for i in nx.connected_components(G)]
    nodes_id = nodes_id[list(compon[idx])]
    W = A[nodes_id[:,None], nodes_id].todense()
    G = graphs.Graph(W)
    G.set_coordinates()
    M = G.N

    yn = signal[nodes_id, :N].T
    xn = signal[:10, :N].T
    ttilde = yn.flatten(order = 'F').reshape(-1,1)

    yn1T = signal[nodes_id, 42:].T
    xn1T = signal[:10, 42:].T
    size = 25
    num_test = 10
    
elif data_name == 'weather':
    coord = scipy.io.loadmat('data/city45data.mat')
    temp = scipy.io.loadmat('data/smhi_temp_17.mat')
    loc = coord['coord45']

    signal = temp['temp_17']

    G = graphs.NNGraph(loc)#, k = 3)

    signal = (signal - np.mean(signal))/np.std(signal)
    data = signal.T

    #N = 30

    np.random.seed(1)
    permute = np.random.permutation(np.arange(95))
    permute = np.delete(permute, np.argwhere(permute==94))
    train = permute[:N]
    test = permute[30:]

    xn = data[:, train].T
    yn = data[:, train+1].T
    ttilde = yn.flatten(order = 'F').reshape(-1,1)

    xn1T = data[:, test].T
    yn1T = data[:, test+1].T

    M = yn.shape[1]
    size = 6
    num_test = 10

# max eigenvalue
#G.compute_laplacian('normalized')
w,_ = np.linalg.eig(G.L.todense())
w = np.real(w).reshape(-1,1)
GL = 1/np.max(w) * np.array(G.L.todense())
w,v = np.linalg.eig(GL)
w = np.real(w).reshape(-1,1)

