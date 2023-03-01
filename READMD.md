# Invariant Representations of Embedded Simplicial Complexes
This repository contains the code used to implement the approach proposed in the paper "Invariant Representations of Embedded Simplicial Complexes".
The approach aims to provide a more robust and invariant method for analyzing geometric objects such as triangular meshes and graphs.

## Abstract
Analyzing embedded simplicial complexes, such as meshes and graphs, is an important problem in many fields.
We propose a new approach for analyzing embedded simplicial complexes in a subdivision-invariant and isometry-invariant way using only topological information.
Our approach is based on creating and analyzing sufficient statistics and based on a graph neural network.
We demonstrate the effectiveness of our approach using a synthetic mesh dataset.

## Getting Started
To get started, clone this repository to your local machine.
Here, we specify the versions of the important Python packages we used:

* Python 3.9.0
* NumPy 1.23.5
* SciPy 1.9.3
* tqdm 4.64.1
* Matplotlib 3.6.2
* PyTorch 1.13.1
* PyTorch Geometric 2.2.0
* PyTorch3D 0.7.2

We recommend creating a virtual environment to manage your dependencies.

## Data
The synthetic mesh data set used in the experiments is not included in this repository due to its size.
Instead, it can be downloaded from [this link](http://people.csail.mit.edu/sumner/research/deftransfer/data.html) and via `download.sh` file.

## Running the code
Once you have the repository and dependencies set up, you can run the code to reproduce the results from the paper.

### Download the data:
```bash
bash download.sh
```
This generate `models` folder containing raw mesh data.

### Preprocess the raw data:
```bash
mkdir preprocessed
python preprocessing.py
```
The preprocessed data that has undergone a random O(3)-transformation and the raw data are saved in the folder `preprocessed`.  
The script `preprocessing.py` contains many hyperparameters, including `N_BINS`, `SUBDIVISION_LEVEL`.
It also generates a file `anim.pth` containing mesh data.
### Train a model:
```bash
mkdir results
python training.py --subdivision_level=7 --n_bins=512 --random_times=1 --training_times=1 
```
It trains a model and save the result in the folder `results`.
### Results:
You can run the file `evaluation_single.py` to see the results.

For the paper, I conducted the experiment using varied configurations and performed it 10 times by setting RANDOM_TIMES==10.
To produce the results in the paper, I use the file `evaluation.py`.

## References
Paik, T. (2023). Invariant Representations of Embedded Simplicial Complexes.
arXiv preprint arXiv:2302.13565. https://arxiv.org/abs/2302.13565
```bibtex
@article{paik2023invariant,
  title={Invariant Representations of Embedded Simplicial Complexes},
  author={Paik, Taejin},
  journal={arXiv preprint arXiv:2302.13565},
  year={2023}
}
```
## License
This repository is licensed under the GNU Lesser General Public License v2.1