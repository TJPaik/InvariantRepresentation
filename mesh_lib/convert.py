import torch


def process(v: torch.FloatTensor, f: torch.IntTensor):
    unique_v, v_inverse = torch.unique(v, dim=0, return_inverse=True)
    f = v_inverse[f]
    f = torch.sort(f, dim=1, )[0]
    f = torch.unique(f, dim=0)
    not_triangle = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2])
    assert not torch.any(not_triangle)
    e = torch.unique(torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]]), dim=0)
    return unique_v, e, f


def triangles2edges(f: torch.IntTensor):
    f = torch.sort(f, dim=1)[0]
    edges = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [0, 2]]])
    edges = torch.unique(edges, dim=0).T
    edges = torch.cat([edges, edges[[1, 0]]], dim=1)
    return edges
