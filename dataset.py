import json 
import os 
from collections import defaultdict 
from typing import Callable, Dict, List, Optional 

import torch 
import argparse

from torch_geometric.data import (
    HeteroData, 
    InMemoryDataset, 
    download_google_url,
    extract_zip
)

# ===== Parse Argument ===== #
parser = argparse.ArgumentParser(description="HGBDataset Loader")
parser.add_argument("--root", type=str, default="../data", help="Root directory for the dataset")
parser.add_argument("--name", type=str, required=True, choices=["acm", "dblp", "freebase", "imdb"], help="Name of the dataset")
parser.add_argument("--force_reload", action="store_true", help="Force reload the dataset")
args = parser.parse_args()


# ===== Define the Heterogeneous Graph Benchmark Dataset ===== #
class HGBDataset(InMemoryDataset):
    r"""An adapted version of HGBDataset from PyTorch Geometric.
    
    A variety of heterogeneous graph benchmark datasets from the
    `"Are We Really Making Much Progress? Revisiting, Benchmarking, and
    Refining Heterogeneous Graph Neural Networks"
    <http://keg.cs.tsinghua.edu.cn/jietang/publications/
    KDD21-Lv-et-al-HeterGNN.pdf>`_ paper.
    
    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (one of :obj:`"ACM"`,
            :obj:`"DBLP"`, :obj:`"Freebase"`, :obj:`"IMDB"`)
        transform (callable, optional): A function/transform that takes in a
            :class:`torch_geometric.data.HeteroData` object and returns a
            transformed version. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in a
            :class:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj: `False`)
    """
    names = {
        "acm": "ACM",
        "dblp": "DBLP",
        "imdb": "IMDB",
        "freebase": "Freebase",
    }

    file_ids = {
        'acm': '1xbJ4QE9pcDJOcALv7dYhHDCPITX2Iddz',
        'dblp': '1fLLoy559V7jJaQ_9mQEsC06VKd6Qd3SC',
        'freebase': '1vw-uqbroJZfFsWpriC1CWbtHCJMGdWJ7',
        'imdb': '18qXmmwKJBrEJxVQaYwKTL3Ny3fPqJeJ2',
    }

    def __init__(
            self, 
            root: str, 
            name: str, 
            transform: Optional[Callable] = None, 
            pre_transform: Optional[Callable] = None, 
            force_reload: bool = False,
    ) -> None: 
        self.name = name.lower() 
        assert self.name in set(self.names.keys()) 
        super().__init__(root, transform, pre_transform, 
                        force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=HeteroData)

    @property 
    def raw_dir(self) -> str: 
        return os.path.join(self.root, self.name, "raw")
    
    @property 
    def processed_dir(self) -> str: 
        return os.path.join(self.root, self.name, "processed") 
    
    @property 
    def raw_file_names(self) -> List[str]:
        x = ["info.dat", "node.dat", "link.dat", "label.dat", "label.dat.test"] 
        return [os.path.join(self.names[self.name], f) for f in x] 
    
    @property 
    def processed_file_names(self) -> str:
        return "data.pt" 
    
    def download(self) -> None: 
        id = self.file_ids[self.name] 
        path = download_google_url(
            id=id, 
            folder=self.raw_dir, 
            filename="data.zip")
        extract_zip(path, self.raw_dir) 
        os.unlink(path) 


    def process(self) -> None: 
        """
        Processes raw data files to create a HeteroData object for graph-based learning tasks.
        This method reads raw data files, extracts node and edge information, and constructs
        a heterogeneous graph representation. It supports datasets for node classification
        and link prediction tasks.

        Supported datasets:
            - Node classification: "acm", "dblp", "freebase", "imdb"
            - Link prediction: Not implemented
        
        Steps:
            1. Parse dataset-specific metadata to extract node and edge types.
            2. Extract node features and create mappings from global to local node indices.
            3. Construct edge indices and edge weights for each edge type.
            4. For node classification tasks, load labels and create train/test masks.

        Raises:
            NotImplementedError: If the dataset is not supported or if link prediction is attempted.

        Attributes:
            self.raw_paths (List[str]): Paths to the raw data files.
            self.processed_paths (List[str]): Paths to save the processed data.
            self.name (str): Name of the dataset.
            self.pre_transform (Callable, optional): A function to apply transformations to the data.

        Outputs:
            Saves the processed HeteroData object to the specified path.
        """
        data = HeteroData() 

        # node_types = {0: "paper", 1: "author", ... }
        # edge_types = {0: ("paper", "cite", "paper"), ...} 
        if self.name in ["acm", "dblp", "imdb"]:

            with open(self.raw_paths[0]) as f: # info.dat
                info = json.load(f) 

            n_types = info["node.dat"]["node type"] 
            n_types = {int(k): v for k, v in n_types.items()} 
            e_types = info["link.dat"]["link type"] 
            e_types = {int(k): tuple(v.values()) for k, v in e_types.items()} 

            for key, (src, dst, rel) in e_types.items():
                src, dst = n_types[int(src)], n_types[int(dst)] 
                rel = rel.split("-")[1] 
                rel = rel if rel != dst and rel[1:] != dst else "to" 
                e_types[key] = (src, rel, dst) 
            
            num_classes = len(info["label.dat"]["node type"]["0"]) 

        elif self.name in ["freebase"]: 

            with open(self.raw_paths[0]) as f: 
                info = f.read().split("\n") 
            
            start = info.index("TYPE\tMEANING") + 1 
            end = info[start:].index("") 
            n_types = [v.split("\t\t") for v in info[start:start + end]] 
            n_types = {int(k): v.lower() for k, v in n_types} 

            e_types = {} 
            start = info.index("LINK\tSTART\tEND\tMEANING") + 1 
            end = info[start:].index("") 
            
            for key, row in enumerate(info[start:start + end]):
                row = row.split("\t")[1:] 
                src, dst, rel = (v for v in row if v != "") 
                src, dst = n_types[int(src)], n_types[int(dst)] 
                rel = rel.split("-")[1] 
                e_types[key] = (src, rel, dst) 
        else: # Link Prediction 
            raise NotImplementedError 
        
        # --- Extract Node Information --- #
        mapping_dict = {} # Map global node indices to local ones 
        x_dict = defaultdict(list) 
        num_nodes_dict: Dict[str, int] = defaultdict(int) 

        with open(self.raw_paths[1], encoding="utf-8", errors="replace") as f: # `node.dat` 
            xs = [v.split("\t") for v in f.read().split("\n")[:-1]] 
        
        for x in xs: 
            n_id, n_type = int(x[0]), n_types[int(x[2])] 
            mapping_dict[n_id] = num_nodes_dict[n_type] 
            num_nodes_dict[n_type] += 1 
            if len(x) >= 4: # Extract feature (in case they are given) 
                x_dict[n_type].append([float(v) for v in x[3].split(",")]) 

        for n_type in n_types.values():
            if len(x_dict[n_type]) == 0:
                data[n_type].num_nodes = num_nodes_dict[n_type] 
            else:
                data[n_type].x = torch.tensor(x_dict[n_type]) 

        edge_index_dict = defaultdict(list) 
        edge_weight_dict = defaultdict(list) 

        with open(self.raw_paths[2]) as f:  # `link.dat` 
            edges = [v.split("\t") for v in f.read().split("\n")[:-1]] 

        for src, dst, rel, weight in edges:
            e_type = e_types[int(rel)] 
            src, dst = mapping_dict[int(src)], mapping_dict[int(dst)] 
            edge_index_dict[e_type].append([src, dst]) 
            edge_weight_dict[e_type].append(float(weight)) 
        
        for e_type in e_types.values(): 
            edge_index = torch.tensor(edge_index_dict[e_type]) 
            edge_weight = torch.tensor(edge_weight_dict[e_type]) 
            data[e_type].edge_index = edge_index.t().contiguous() 
            # Only add "weighted" edges to the graph 
            if not torch.allclose(edge_weight, torch.ones_like(edge_weight)):
                data[e_type].edge_weight = edge_weight 

        # --- Node Classification --- #
        if self.name in ["acm", "dblp", "freebase", "imdb"]: 

            with open(self.raw_paths[3]) as f: # `label.dat` 
                train_ys = [v.split("\t") for v in f.read().split("\n")[:-1]]

            with open(self.raw_paths[4]) as f: # `label.dat.test` 
                test_ys = [v.split("\t") for v in f.read().split("\n")[:-1]] 
                
            for y in train_ys: 
                n_id, n_type = mapping_dict[int(y[0])], n_types[int(y[2])] 

                if not hasattr(data[n_type], "y"): 
                    num_nodes = data[n_type].num_nodes 

                    if self.name in ["imdb"]: # multi-label 
                        data[n_type].y = torch.zeros((num_nodes, num_classes)) 
                    else:
                        data[n_type].y = torch.full((num_nodes, ), -1).long() 
                    
                    data[n_type].train_mask = torch.zeros(num_nodes).bool()
                    data[n_type].test_mask = torch.zeros(num_nodes).bool() 

                if data[n_type].y.dim() > 1: # multi-label
                    for v in y[3].split(","): 
                        data[n_type].y[n_id, int(v)] = 1 
                else: 
                    data[n_type].y[n_id] = int(y[3]) 

                data[n_type].train_mask[n_id] = True 
            
            for y in test_ys: 
                n_id, n_type = mapping_dict[int(y[0])], n_types[int(y[2])]

                if data[n_type].y.dim() > 1: 
                    for v in y[3].split(","):
                        data[n_type].y[n_id, int(v)] = 1
                else: 
                    data[n_type].y[n_id] = int(y[3])

                data[n_type].test_mask[n_id] = True

        else: # Link Prediction 
            raise NotImplementedError
        
        if self.pre_transform is not None: 
            data = self.pre_transform(data) 

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str: 
        return f"{self.names[self.name]}()" 

# ===== Main Execution ===== #
if __name__ == "__main__":
    dataset = HGBDataset(
        root=args.root, 
        name=args.name, 
        force_reload=args.force_reload)
    print(dataset)
    print(dataset[0])
    print(dataset[0].edge_index_dict)
    print(dataset[0].num_nodes_dict)