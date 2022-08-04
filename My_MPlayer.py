import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.inits import reset, uniform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GCNConv(MessagePassing):
    """
    referred: https://pytorch-geometric.readthedocs.io/en/1.5.0/_modules/torch_geometric/nn/conv/nn_conv.html
    """
    def __init__(self, in_channels, out_channels, nn):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.bias = Parameter(torch.Tensor(out_channels))
        self.nn = nn
        self.reset_parameters()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index, edge_attr):

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_j, pseudo):
        weight = self.nn(pseudo).view(-1, self.in_channels, self.out_channels)
        return torch.matmul(x_j.unsqueeze(1), weight).squeeze(1)

    def update(self, aggr_out, x):
        aggr_out = aggr_out + self.bias
        return aggr_out