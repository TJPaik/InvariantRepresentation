import os
from pathlib import Path

import numpy as np
import psutil
import torch
from pytorch3d.io import load_objs_as_meshes
from scipy.stats import ortho_group
from tqdm import tqdm

from mesh_lib import icosphere
from mesh_lib import process

RANGE = 8.0
N_BINS = 512
SUBDIVISION_LEVEL = 7
RANDOM_TIMES = 1
INITIAL_RANDOM_SEED = 42

data_torch_path = 'anim.pth'
preprocessed_file_name = f'preprocessed/{RANGE}_{N_BINS}_{SUBDIVISION_LEVEL}_{INITIAL_RANDOM_SEED}'


def generate_file(random: int):
    euler_curves = []
    for r, (v, e, f) in tqdm(enumerate(vefs), total=len(vefs)):
        v = v / v.std()
        if random != -1:
            torch.manual_seed(INITIAL_RANDOM_SEED + random + r)
            v = (v @ ortho_group.rvs(3, random_state=INITIAL_RANDOM_SEED + random + r)) + torch.randn(3)
        assert (-RANGE < v.min()) & (RANGE > v.max())
        ips = (v @ directions.T).float()
        evm = torch.max(ips[e], axis=1)[0].float()
        fvm = torch.max(ips[f], axis=1)[0].float()
        vh = torch.stack([torch.histc(ips[:, i], N_BINS, -RANGE, RANGE).int() for i in range(len(directions))])
        eh = torch.stack([torch.histc(evm[:, i], N_BINS, -RANGE, RANGE).int() for i in range(len(directions))])
        fh = torch.stack([torch.histc(fvm[:, i], N_BINS, -RANGE, RANGE).int() for i in range(len(directions))])
        euler = vh + fh - eh
        euler = torch.cumsum(euler, dim=1)
        assert len(euler[:, -1].unique()) == 1
        euler_curves.append(euler)
    euler_curves = torch.stack(euler_curves)
    return euler_curves


if __name__ == '__main__':
    torch.set_num_threads(psutil.cpu_count())

    if not os.path.exists(data_torch_path):
        mesh_data_folder = Path('models')
        obj_paths = [el.as_posix() for el in mesh_data_folder.glob('**/*.obj')]
        obj_paths = [el for el in obj_paths if 'collapse' not in el]
        obj_paths = sorted(obj_paths)
        labels = np.asarray([el.split('/')[-2].split('-')[-2] for el in obj_paths])
        assert len(obj_paths) == 229 and len(labels) == 229

        meshes = load_objs_as_meshes(obj_paths, load_textures=False)
        vefs = [process(v, f) for v, f in tqdm(zip(meshes.verts_list(), meshes.faces_list()), total=len(meshes))]
        torch.save([vefs, labels], data_torch_path)
    else:
        vefs, labels = torch.load(data_torch_path)

    directions = icosphere(SUBDIVISION_LEVEL)[0]

    result = generate_file(-1)
    torch.save(result, f'{preprocessed_file_name}_without_transform.pth')

    for i in range(RANDOM_TIMES):
        result = generate_file(i)
        torch.save(result, f'{preprocessed_file_name}_{i}.pth')
