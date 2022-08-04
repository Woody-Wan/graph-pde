import numpy as np
import torch
import sklearn.metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MeshGen(object):
    def __init__(self, grid_size, mesh_size, sample_size):
        super(MeshGen, self).__init__()

        self.d = len(grid_size)
        self.s = mesh_size[0]
        self.m = sample_size

        grids = []
        self.n = 1
        for j in range(self.d):
            grids.append(np.linspace(grid_size[j][0], grid_size[j][1], mesh_size[j]))
            self.n *= mesh_size[j] # num of index
        self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T  # generate mesh grid
        self.idx = np.array(range(self.n))
        self.grid_sample = self.grid

    def index(self, r):
        adj_matrix = sklearn.metrics.pairwise_distances(self.grid_sample)  # generate adjacent matrix
        self.edge_index = np.vstack(np.where(adj_matrix <= r))  # qualified index
        self.n_edges = self.edge_index.shape[1]  # calculate the number of qualified index

        return torch.tensor(self.edge_index, dtype=torch.long)

    def attribute(self, data):
        if data is None:
            edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
        else:
            edge_attr = np.zeros((self.n_edges, 3 * self.d))
            edge_attr[:, 0:2 * self.d] = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
            edge_attr[:, 2 * self.d] = data[self.edge_index[0]]
            edge_attr[:, 2 * self.d + 1] = data[self.edge_index[1]]

        return torch.tensor(edge_attr, dtype=torch.float)

    def sample(self):
        perm = torch.randperm(self.n)
        self.idx = perm[:self.m]
        self.grid_sample = self.grid[self.idx]
        return self.idx

    def get_grid(self):
        return torch.tensor(self.grid_sample, dtype=torch.float)