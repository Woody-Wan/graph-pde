import numpy as np
import scipy.io
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch_geometric.nn as geom_nn

from torch_geometric.data import Data, DataLoader

from My_mesh_grid import *
from My_normalization import *
from My_Metrics import *
from My_MPlayer import *
from My_kernel import *
from utilities import *

TRAIN_PATH = 'data/piececonst_r241_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r241_N1024_smooth2.mat'
'''
train_mat = scipy.io.loadmat(TRAIN_PATH)
test_mat = scipy.io.loadmat(TEST_PATH)
'''
# parameters
r = 8
s = int(((241 - 1) / r) + 1)
n = s ** 2
m = 50
k = 5

radius_train = 0.15
radius_test = 0.15

print('resolution', s)

ntrain = 100
ntest = 100

batch_size = 1
batch_size2 = 2
width = 64
ker_width = 512
depth = 3
edge_features = 6
node_features = 6

epochs = 200
learning_rate = 0.0001
scheduler_step = 50
scheduler_gamma = 0.5

# save path
path = 'UAIMine_r' + str(s) + '_n' + str(ntrain)
path_model = 'model/' + path + ''
path_train_err = 'results/' + path + 'train.txt'
path_test_err = 'results/' + path + 'test.txt'
path_image = 'image/' + path + ''
path_train_err = 'results/' + path + 'train'
path_test_err = 'results/' + path + 'test'
path_image_train = 'image/' + path + 'train'
path_image_test = 'image/' + path + 'test'

# read files

train_a = torch.from_numpy(scipy.io.loadmat(TRAIN_PATH)['coeff'][:ntrain, ::r, ::r].reshape(ntrain, -1))
train_a_smooth = torch.from_numpy(scipy.io.loadmat(TRAIN_PATH)['Kcoeff'][:ntrain, ::r, ::r].reshape(ntrain, -1))
train_a_gradx = torch.from_numpy(scipy.io.loadmat(TRAIN_PATH)['Kcoeff_x'][:ntrain, ::r, ::r].reshape(ntrain, -1))
train_a_grady = torch.from_numpy(scipy.io.loadmat(TRAIN_PATH)['Kcoeff_y'][:ntrain, ::r, ::r].reshape(ntrain, -1))
train_u = torch.from_numpy(scipy.io.loadmat(TRAIN_PATH)['sol'][:ntrain, ::r, ::r].reshape(ntrain, -1))

test_a = torch.from_numpy(scipy.io.loadmat(TEST_PATH)['coeff'][:ntest, ::r, ::4].reshape(ntest, -1))
test_a_smooth = torch.from_numpy(scipy.io.loadmat(TEST_PATH)['Kcoeff'][:ntest, ::r, ::4].reshape(ntest, -1))
test_a_gradx = torch.from_numpy(scipy.io.loadmat(TEST_PATH)['Kcoeff_x'][:ntest, ::r, ::4].reshape(ntest, -1))
test_a_grady = torch.from_numpy(scipy.io.loadmat(TEST_PATH)['Kcoeff_y'][:ntest, ::r, ::4].reshape(ntest, -1))
test_u = torch.from_numpy(scipy.io.loadmat(TEST_PATH)['sol'][:ntest, ::r, ::4].reshape(ntest, -1))
'''

reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('coeff')[:ntrain, ::r, ::r].reshape(ntrain, -1)
train_a_smooth = reader.read_field('Kcoeff')[:ntrain, ::r, ::r].reshape(ntrain, -1)
train_a_gradx = reader.read_field('Kcoeff_x')[:ntrain, ::r, ::r].reshape(ntrain, -1)
train_a_grady = reader.read_field('Kcoeff_y')[:ntrain, ::r, ::r].reshape(ntrain, -1)
train_u = reader.read_field('sol')[:ntrain, ::r, ::r].reshape(ntrain, -1)

reader.load_file(TEST_PATH)
test_a = reader.read_field('coeff')[:ntest, ::r, ::r].reshape(ntest, -1)
test_a_smooth = reader.read_field('Kcoeff')[:ntest, ::r, ::r].reshape(ntest, -1)
test_a_gradx = reader.read_field('Kcoeff_x')[:ntest, ::r, ::r].reshape(ntest, -1)
test_a_grady = reader.read_field('Kcoeff_y')[:ntest, ::r, ::r].reshape(ntest, -1)
test_u = reader.read_field('sol')[:ntest, ::r, ::r].reshape(ntest, -1)
'''
# normalization
Normalizer = Normalization(train_a)
train_a = Normalizer.encode(train_a)
test_a = Normalizer.encode(test_a)
as_normalizer = Normalization(train_a_smooth)
train_a_smooth = as_normalizer.encode(train_a_smooth)
test_a_smooth = as_normalizer.encode(test_a_smooth)
agx_normalizer = Normalization(train_a_gradx)
train_a_gradx = agx_normalizer.encode(train_a_gradx)
test_a_gradx = agx_normalizer.encode(test_a_gradx)
agy_normalizer = Normalization(train_a_grady)
train_a_grady = agy_normalizer.encode(train_a_grady)
test_a_grady = agy_normalizer.encode(test_a_grady)

u_normalizer = Unit_Normalization(train_u)
train_u = u_normalizer.encode(train_u)

# data generalization
mesh_generator = MeshGen([[0, 1], [0, 1]], [s, s], sample_size=m)
grid = mesh_generator.get_grid()

data_train = []

for j in range(ntrain):
    for i in range(k):
        idx = mesh_generator.sample()
        grid = mesh_generator.get_grid()
        edge_index = mesh_generator.index(radius_train)
        edge_attr = mesh_generator.attribute(data=train_a[j, :])

        data_train.append(Data(x=torch.cat([grid, train_a[j, idx].reshape(-1, 1),
                                            train_a_smooth[j, idx].reshape(-1, 1), train_a_gradx[j, idx].reshape(-1, 1),
                                            train_a_grady[j, idx].reshape(-1, 1)
                                            ], dim=1),
                               y=train_u[j, idx],
                               edge_index=edge_index, edge_attr=edge_attr, sample_size=idx
                               ))

mesh_generator = MeshGen([[0, 1], [0, 1]], [s, s], sample_size=m)
grid = mesh_generator.get_grid()

data_test = []

for j in range(ntest):
    for i in range(k):
        idx = mesh_generator.sample()
        grid = mesh_generator.get_grid()
        edge_index = mesh_generator.index(radius_test)
        edge_attr = mesh_generator.attribute(data=test_a[j, :])

        data_test.append(Data(x=torch.cat([grid, test_a[j, idx].reshape(-1, 1),
                                           test_a_smooth[j, idx].reshape(-1, 1),
                                           test_a_gradx[j, idx].reshape(-1, 1),
                                           test_a_grady[j, idx].reshape(-1, 1)
                                           ], dim=1),
                              y=test_u[j, idx],
                              edge_index=edge_index, edge_attr=edge_attr, sample_size=idx
                              ))

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)


# Model architecture
class kernelModel(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1):
        super(kernelModel, self).__init__()

        self.depth = depth
        self.fc1 = torch.nn.Linear(in_width, width)

        kernel = MLPModel([ker_in, ker_width // 2, ker_width, width ** 2])
        self.GCN = GCNConv(width, width, kernel, aggr='mean')
        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.float()
        # edge_index = edge_index.long()
        # edge_attr = edge_attr.float()
        x = self.fc1(x)
        for d in range(self.depth):
            x = torch.nn.functional.relu(self.GCN(x, edge_index, edge_attr))
        x = self.fc2(x)
        return x


# training
# device = torch.device('cuda')

model = kernelModel(width, ker_width, depth, edge_features, node_features).cpu()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
u_normalizer.cpu()

model.train()
ttrain = np.zeros((epochs,))
ttest = np.zeros((epochs,))

for ep in range(epochs):
    train_mse = 0.0
    train_l2 = 0.0
    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)
        mse = torch.nn.functional.mse_loss(out.view(-1, 1), batch.y.view(-1, 1))
        mse.backward()

        l2 = myloss(
            u_normalizer.decode(out.view(batch_size, -1), sample_idx=batch.sample_idx.view(batch_size, -1)),
            u_normalizer.decode(batch.y.view(batch_size, -1), sample_idx=batch.sample_idx.view(batch_size, -1)))
        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()

    model.eval()

    ttrain[ep] = train_l2 / (ntrain * k)

    print(ep, ' train_mse:', train_mse / len(train_loader))

u_normalizer
model = model
test_l2 = 0.0
with torch.no_grad():
    for batch in test_loader:
        out = model(batch)
        test_l2 += myloss(u_normalizer.decode(out.view(batch_size2, -1)),
                          batch.y.view(batch_size2, -1))

ttest[ep] = test_l2 / ntest

print(' train_mse:', train_mse / len(train_loader),
      ' test:', test_l2 / ntest)
np.savetxt(path_train_err + '.txt', ttrain)
np.savetxt(path_test_err + '.txt', ttest)

torch.save(model, path_model)

# plot

resolution = s
data = train_loader.dataset[0]
coeff = data.coeff.numpy().reshape((resolution, resolution))
truth = u_normalizer.decode(data.y.reshape(1, -1)).numpy().reshape((resolution, resolution))
approx = u_normalizer.decode(model(data).reshape(1, -1)).detach().numpy().reshape((resolution, resolution))
_min = np.min(np.min(truth))
_max = np.max(np.max(truth))

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(truth, vmin=_min, vmax=_max)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Ground Truth')

plt.subplot(1, 3, 2)
plt.imshow(approx, vmin=_min, vmax=_max)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Approximation')

plt.subplot(1, 3, 3)
plt.imshow((approx - truth) ** 2)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Error')

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig(path_image_train + '.png')

resolution = m
data = test_loader.dataset[0]
coeff = data.coeff.numpy().reshape((resolution, resolution))
truth = data.y.numpy().reshape((resolution, resolution))
approx = u_normalizer.decode(model(data).reshape(1, -1)).detach().numpy().reshape((resolution, resolution))
_min = np.min(np.min(truth))
_max = np.max(np.max(truth))

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(truth, vmin=_min, vmax=_max)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Ground Truth')

plt.subplot(1, 3, 2)
plt.imshow(approx, vmin=_min, vmax=_max)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Approximation')

plt.subplot(1, 3, 3)
plt.imshow((approx - truth) ** 2)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Error')

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.savefig(path_image_test + '.png')
