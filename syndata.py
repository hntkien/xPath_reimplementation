import os 
import pandas as pd 
import torch 
import dgl 
import numpy as np 
from dgl.data import DGLDataset 
# # from dgl.heterograph import DGLHeteroGraph 
# from typing import Tuple, Dict 

class SyntheticHGBDataset(DGLDataset):
    """Custom DGL Dataset for Synthetic Heterogeneous Graphs.

    This class loads and processes a synthetic heterogeneous graph dataset from CSV files. It saves the processed graph for reuse. 

    Attributes: 
        dataset_name (str): Name of the dataset folder. 
        raw_dir (str): Path to the parent directory containing all datasets. 

    Args:
        dataset_name (str): Name of the dataset folder. 
        raw_dir (str): Parent directory containing the dataset folders. 

    Returns: 
        graph (DGLHeteroGraph): The constructed heterogeneous graph.
        num_classes (int): Number of classes in the dataset.
    """
    def __init__(self, dataset_name: str, raw_dir: str="./data"):
        self.dataset_name = dataset_name 
        self.dataset_path = os.path.join(raw_dir, dataset_name) 
        super().__init__(name=dataset_name) 

    def process(self): 
        # --- Load Node Data --- #
        node_df = pd.read_csv(
            os.path.join(self.dataset_path, "node.csv"), 
            header=0) 
        node_df.columns = ["nid", "ntype", "feat"] 
        node_types = node_df["ntype"].unique() 
        node_dict = {str(ntype): [] for ntype in node_types} 
        features_dict = {} 

        for _, row in node_df.iterrows():
            node_dict[str(row["ntype"])].append(row["nid"]) 

        # --- Map original node IDs to local IDs per node type --- #
        id_maps = {
            str(ntype): {nid: i for i, nid in enumerate(node_dict[str(ntype)])}
            for ntype in node_types
        }

        # --- Parse Features --- #
        for ntype in node_types:
            ntype_df = node_df[node_df["ntype"] == ntype] 
            feats = ntype_df["feat"].apply(lambda x: torch.tensor(
                [float(i) for i in x.split(" ")], dtype=torch.float)) 
            features_dict[str(ntype)] = torch.stack(feats.tolist()) 

        # --- Load Edge Data --- #
        edge_df = pd.read_csv(
            os.path.join(self.dataset_path, "link.csv"), 
            header=0)
        edge_df.columns = ["src", "dst", "etype", "weight"] 
        edge_types = edge_df["etype"].unique() 

        edges = {} 

        # --- Create a mapping from node ID to node type --- # 
        nid_to_ntype = dict(zip(node_df["nid"], node_df["ntype"])) 

        for _, row in edge_df.iterrows():
            src_id = row["src"] 
            dst_id = row["dst"]
            etype_id = int(row["etype"]) 

            src_type = str(nid_to_ntype[src_id]) 
            dst_type = str(nid_to_ntype[dst_id]) 
            rel_type = f"rel_{etype_id}" 

            canonical_etype = (src_type, rel_type, dst_type) 

            # Initialise if not exist 
            if canonical_etype not in edges: 
                edges[canonical_etype] = ([], []) 

            # Append the source and destination node IDs to the list
            edges[canonical_etype][0].append(id_maps[src_type][src_id])
            edges[canonical_etype][1].append(id_maps[dst_type][dst_id])

        # --- Convert Edge Lists to Tensors --- # 
        for key in edges: 
            edges[key] = (
                torch.tensor(edges[key][0]), torch.tensor(edges[key][1]))
            
        # --- Create the Heterogeneous Graph --- # 
        self.graph = dgl.heterograph(edges) 

        # --- Assign Node Features --- # 
        for ntype in self.graph.ntypes: 
            self.graph.nodes[ntype].data["feat"] = features_dict[ntype] 

        # --- Load labels and splits --- #
        self.label_ntype = node_df[
            node_df["nid"].isin(pd.read_csv(os.path.join(self.dataset_path, "labels.csv"), header=None)[0])]["ntype"].iloc[0]
        self.label_ntype = str(self.label_ntype) 
        self.graph.nodes[self.label_ntype].data["label"] = self._load_labels("labels.csv") 

        for split in ["train", "val", "test"]: 
            mask = torch.zeros(
                self.graph.num_nodes(self.label_ntype), 
                dtype=torch.bool)
            ids = pd.read_csv(
                os.path.join(self.dataset_path, f"data_{split}.csv"), 
                header=None)[0].tolist()
            mapped_ids = [id_maps[self.label_ntype][i] for i in ids] 
            mask[mapped_ids] = True 
            self.graph.nodes[self.label_ntype].data[f"{split}_mask"] = mask 

        # --- Save number of classes --- #
        self.num_classes = int(
            self.graph.nodes[self.label_ntype].data["label"].max().item() + 1)
        
    def _load_labels(self, filename: str) -> torch.Tensor: 
        df = pd.read_csv(
            os.path.join(self.dataset_path, filename), 
            header=0
        )
        df.columns = ["nid", "ntype", "label"]
        df = df[df["ntype"] == int(self.label_ntype)] 
        # df = df.sort_values("nid") 
        # labels = df["label"].tolist() 
        # return torch.tensor(labels, dtype=torch.long) 

        label_strs = df["label"].astype(str).tolist() 

        # Check for multi-labels 
        is_multi_label = any(" " in label for label in label_strs) 

        if is_multi_label:
            label_lists = [
                list(map(int, label.split())) for label in label_strs]
            max_label = max(
                [max(lbls) for lbls in label_lists if lbls]) if label_lists else 0 
            label_tensor = torch.zeros(
                (len(label_lists), max_label+1), dtype=torch.float32)
            
            for i, lbls in enumerate(label_lists):
                label_tensor[i, lbls] = 1.0 
            
            self.is_multi_label = True 
            return label_tensor 
        else:
            labels = torch.tensor(
                [int(label) for label in label_strs], dtype=torch.long) 
            self.is_multi_label = False 
            return labels
        
    def has_cache(self) -> bool:
        return os.path.exists(self.save_path) 
    
    def save(self):
        dgl.save_graphs(
            self.save_path, 
            [self.graph], 
            {"num_classes": torch.tensor([self.num_classes])})
        
    def load(self): 
        graphs, label_dict = dgl.load_graphs(self.save_path) 
        self.graph = graphs[0]
        self.num_classes = label_dict["num_classes"].item() 

    @property 
    def save_path(self): 
        return os.path.join(
            self.dataset_path, f"{self.dataset_name}_processed_graph.bin")
    
    def __getitem__(self, idx: int):
        assert idx == 0, "Only one graph in this dataset."
        return self.graph 
    
    def __len__(self) -> int:
        return 1

if __name__ == "__main__":
    dataset = SyntheticHGBDataset(dataset_name="syn_acm")
    graph = dataset[0]
    print(graph)
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Label node type: {dataset.label_ntype}")
    print(f"Number of nodes: {graph.num_nodes()}")
    print(f"Number of edges: {graph.num_edges()}")
    print(f"Node types: {graph.ntypes}")
    print(f"Edge types: {graph.etypes}")
    for ntype in graph.ntypes:
        print(f"Number of nodes of type '{ntype}': {graph.num_nodes(ntype)}")
    for etype in graph.etypes:
        print(f"Number of edges of type '{etype}': {graph.num_edges(etype)}")