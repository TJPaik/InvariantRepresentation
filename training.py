import argparse

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from SCN import Model
from mesh_lib import icosphere
from mesh_lib import triangles2edges

device = 'cuda'


def train(trial_n, args):
    SUBDIVISION_LEVEL = args.subdivision_level
    N_BINS = args.n_bins
    RANDOM_TIMES = args.random_times
    N_SAMPLES_FOR_TRAINING = args.n_samples_for_training
    N_EPOCHS = args.n_epochs
    N_BATCH = args.n_batch
    INITIAL_RANDOM_SEED = args.initial_random_seed
    print(trial_n, SUBDIVISION_LEVEL, N_BINS, RANDOM_TIMES, N_SAMPLES_FOR_TRAINING, N_EPOCHS, N_BATCH,
          INITIAL_RANDOM_SEED)

    # Preparation of training data
    euler_data = torch.load(f'preprocessed/8.0_{N_BINS}_{SUBDIVISION_LEVEL}_42_without_transform.pth')
    _, labels = torch.load('anim.pth')
    unique_labels = np.unique(labels)

    np.random.seed(INITIAL_RANDOM_SEED + trial_n)
    train_idx = np.concatenate(
        [np.random.choice(np.where(labels == el)[0], N_SAMPLES_FOR_TRAINING, replace=False).tolist()
         for el in np.sort(unique_labels)])

    tmp = torch.from_numpy(np.concatenate([[i] * N_SAMPLES_FOR_TRAINING for i in range(len(unique_labels))]))
    training_reps = torch.stack([
        torch.cos(tmp / len(unique_labels) * 2 * torch.pi),
        torch.sin(tmp / len(unique_labels) * 2 * torch.pi),
    ]).T

    ds = TensorDataset(euler_data[train_idx], training_reps)
    dl = DataLoader(ds, N_BATCH, shuffle=True, drop_last=False)

    # Model
    _, tris = icosphere(SUBDIVISION_LEVEL)
    edges = triangles2edges(torch.from_numpy(tris))
    torch.manual_seed(INITIAL_RANDOM_SEED + trial_n)
    model = Model(edges)
    model.to(device)

    # Optimize
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    optimizer = optim.Adam(model.parameters(), 0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 200, 0.1)
    model.train()
    tbar = tqdm(range(N_EPOCHS))
    for _ in tbar:
        for el1, el2 in dl:
            tmp_output, _ = model(el1.to(device).float())
            loss = loss_fn(tmp_output, el2.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        tbar.set_postfix_str(f"{loss.item()}_{scheduler.get_last_lr()}")

    # Testing
    output = []
    for i in tqdm(range(RANDOM_TIMES)):
        testing_euler_data = torch.load(f'preprocessed/8.0_{N_BINS}_{SUBDIVISION_LEVEL}_42_{i}.pth')
        ds_testing = TensorDataset(testing_euler_data, )
        dl_testing = DataLoader(ds_testing, N_BATCH, shuffle=False, drop_last=False)
        model.eval()
        tmp_output = []
        with torch.no_grad():
            for el, in dl_testing:
                tmp_output.append(
                    model(el.to(device).float())[0].detach().cpu()
                )
        tmp_output = torch.cat(tmp_output)
        output.append(tmp_output)
    output = torch.stack(output)
    torch.save(output, f'results/8.0_{N_BINS}_{SUBDIVISION_LEVEL}_42_{trial_n}.pth')


def get_args():
    parser = argparse.ArgumentParser(description='Train a model with ANIM dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--subdivision_level', dest='subdivision_level', type=int)
    parser.add_argument('--n_bins', dest='n_bins', type=int)
    parser.add_argument('--random_times', dest='random_times', type=int, default=10)
    parser.add_argument('--n_samples_for_training', dest='n_samples_for_training', type=int, default=5)
    parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=400)
    parser.add_argument('--n_batch', dest='n_batch', type=int, default=16)
    parser.add_argument('--training_times', dest='training_times', type=int, default=8)
    parser.add_argument('--initial_random_seed', dest='initial_random_seed', type=int, default=42)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    TRAINING_TIMES = args.training_times
    for t in range(TRAINING_TIMES):
        train(t, args)
