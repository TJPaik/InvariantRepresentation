import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == '__main__':
    result = torch.load('results/8.0_512_7_42_0.pth').view(-1, 2)
    _, labels = torch.load('anim.pth')
    plt.figure(figsize=(7, 7))
    for el in np.unique(labels):
        plt.scatter(*(result[el == labels]).T, label=el, s=70)
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()
    plt.close()
