from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == '__main__':
    path = Path('results/')
    results1 = torch.stack([torch.stack([torch.load(el2.as_posix()) for el2 in path.glob(el)]) for el in
                            [f"8.0_512_{i}_42_*.pth" for i in [1, 4, 7, 10, 13]]])
    results2 = torch.stack([torch.stack([torch.load(el2.as_posix()) for el2 in path.glob(el)]) for el in
                            [f"8.0_{i}_10_42_*.pth" for i in [32, 64, 128, 256, 512, 1024]]])

    mean_points1 = torch.mean(results1, dim=2)[:, :, None, ...]
    mean_points2 = torch.mean(results2, dim=2)[:, :, None, ...]
    diff_from_mean1 = torch.linalg.norm(results1 - mean_points1, dim=4)
    diff_from_mean2 = torch.linalg.norm(results2 - mean_points2, dim=4)

    stds1 = torch.mean(diff_from_mean1, dim=[1, 2, 3])
    stds2 = torch.mean(diff_from_mean2, dim=[1, 2, 3])
    print(stds1, stds2)

    plt.rcParams['font.size'] = 20
    _, labels = torch.load('anim.pth')
    for i in range(3):
        plt.figure(figsize=(7, 7))
        for el in np.unique(labels):
            plt.scatter(*(results1[-1][0][i][el == labels]).T, label=el, s=70)
        plt.legend()
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"figures/512_10_trial_{i}.svg")
        plt.show()
        plt.close()
