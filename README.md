# xPath
This is a [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/index.html) reimplementation for the paper: Towards Fine-grained Explainability for Heterogeneous Graph Neural Network.


## Requirements
- Python 3.10
- torch~=2.5.0
- PyG~=2.6.1
- numpy~=2.1.2
- tqdm~=4.67.1


## Datasets

The heterogeneous graph datasets we use are [DBLP](https://github.com/BUPT-GAMMA/HeCo/tree/main/data/dblp), [ACM](https://github.com/BUPT-GAMMA/HeCo/tree/main/data/acm), and [IMDB](https://www.kaggle.com/carolzhangdc/imdb-5000-movie-dataset). 
Datasets are provided in the `data` folder. Taking DBLP dataset as an example:
- `dblp_graph.bin`: Heterogeneous graph。
- `dblp_index_60.bin`: Training, validation, and test set for SimpleHGN.
- `dblp_index_2000.bin`: Training, validation, and test set for HGT.

## Train HGNs

Trained HGNs are provided in `ckpt/{dataset_name}/bk` for reproducing the results in our paper. To retrain the HGNs, run

```shell
# edit configurations in config.py
python train.py
```

## Generate explanations

```shell
# edit configurations in config.py
python main.py
```

