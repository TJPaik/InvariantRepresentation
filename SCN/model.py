import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn.conv import SGConv


class Model(nn.Module):
    def __init__(self, edge_indices=torch.IntTensor):
        super(Model, self).__init__()
        self.register_buffer("EI", edge_indices)

        self.conv1 = nn.Conv1d(1, 128, kernel_size=5, stride=2, bias=True)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=5, stride=2, bias=True)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=5, stride=2, bias=True)
        self.gconv1 = SGConv(128, 128, 39, cached=False, add_self_loops=True)
        self.gconv2 = SGConv(128, 128, 39, cached=False, add_self_loops=True)
        self.linear = nn.Linear(128, 2)

        self.batch_edges = torch.Tensor
        self.n_batch = 0

    def forward(self, x):
        x_shape = x.shape
        if x_shape[0] != self.n_batch:
            self.batch_edges = torch.cat([self.EI + (x_shape[1] * i) for i in range(x_shape[0])], dim=1)
            self.n_batch = x_shape[0]

        x = x.view(x_shape[0] * x_shape[1], 1, -1)

        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.adaptive_max_pool1d(x, 1)[..., 0]

        y = x

        x = self.gconv1(x, self.batch_edges)
        x = F.leaky_relu(x)

        x = self.gconv2(x, self.batch_edges)
        x = F.leaky_relu(x)

        x = x.view(x_shape[0], x_shape[1], -1)
        x = torch.mean(x, dim=1)

        x = self.linear(x)

        return x, y


if __name__ == '__main__':
    import os
    import sys

    os.chdir('../')
    sys.path.append('.')
    from mesh_lib import icosphere
    from mesh_lib import triangles2edges

    directions, tris = icosphere(1)
    edges = triangles2edges(torch.from_numpy(tris))

    dummy_input = torch.randn(8, len(directions), 500)
    model = Model(edges)
    model.eval()

    with torch.no_grad():
        output = model(dummy_input)
    print(output[0].shape, output[1].shape)
