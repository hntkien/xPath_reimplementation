import os 
import pandas as pd 
import torch 
import dgl 
from dgl.data import DGLDataset 

# ========== SyntheticHGBDataset Class ========== #
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
    def __init__(
            self, 
            dataset_name: str, 
            raw_dir: str="./data", 
            force_reload: bool=False):
        self.dataset_name = dataset_name 
        self.dataset_path = os.path.join(raw_dir, dataset_name) 
        super().__init__(name=dataset_name, force_reload=force_reload) 

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
        # self.num_classes = int(
        #     self.graph.nodes[self.label_ntype].data["label"].max().item() + 1)
        label_data = self.graph.nodes[self.label_ntype].data["label"] 

        if self.is_multi_label: 
            self.num_classes = label_data.shape[1] 
        else:
            self.num_classes = int(label_data.max().item()) + 1 

        
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
    
# ========== End of SyntheticHGBDataset Class ========== #

# ========== Check Dataset Conditions ========== # 
def check_dataset_conditions(graph, target_ntype, is_multi_label):
    node_data = graph.nodes[target_ntype].data
    label = node_data['label']
    train_mask = node_data.get('train_mask')
    val_mask = node_data.get('val_mask')
    test_mask = node_data.get('test_mask')

    # Check if masks are present
    assert train_mask is not None, "'train_mask' not found"
    assert val_mask is not None, "'val_mask' not found"
    assert test_mask is not None, "'test_mask' not found"

    # Check shape and dtype of masks
    for name, mask in [('train_mask', train_mask), ('val_mask', val_mask), ('test_mask', test_mask)]:
        assert mask.dtype == torch.bool, f"'{name}' must be a boolean tensor"
        assert mask.ndim == 1, f"'{name}' must be a 1D tensor"
        assert mask.shape[0] == graph.num_nodes(target_ntype), f"'{name}' must match number of nodes of type '{target_ntype}'"

    # Check label shape
    if is_multi_label:
        assert label.ndim == 2, "'label' should be 2D for multi-label classification"
    else:
        assert label.ndim == 1 or (label.ndim == 2 and label.shape[1] == 1), "'label' should be 1D for single-label classification"

    print("✅ All dataset conditions passed.")


# ========== Main Execution ========== #
if __name__ == "__main__":
    dataset = SyntheticHGBDataset(dataset_name="syn_recipe", force_reload=True)
    graph = dataset[0]

    print(graph)
    print(f"\n Dataset Statistics")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Is multi-label: {dataset.is_multi_label}")
    print(f"Total number of nodes: {graph.num_nodes()}")
    print(f"Total number of edges: {graph.num_edges()}")
    print(f"Node types: {graph.ntypes}")
    print(f"Edge types: {graph.etypes}")

    print("\n Node Type Summary")
    for ntype in graph.ntypes:
        print(f"  - '{ntype}': {graph.num_nodes(ntype)} nodes")
        if 'label' in graph.nodes[ntype].data:
            labels = graph.nodes[ntype].data['label']
            print(f"    • Labels present with shape: {labels.shape}")
            if not dataset.is_multi_label:
                unique_labels, counts = torch.unique(labels, return_counts=True)
                print(f"    • Label distribution: {dict(zip(unique_labels.tolist(), counts.tolist()))}")

        for mask_type in ['train_mask', 'val_mask', 'test_mask']:
            if mask_type in graph.nodes[ntype].data:
                num_masked = int(graph.nodes[ntype].data[mask_type].sum().item())
                print(f"    • {mask_type}: {num_masked} nodes")

    print("\n Edge Type Summary")
    for etype in graph.canonical_etypes:
        print(f"  - '{etype}': {graph.num_edges(etype)} edges")

    check_dataset_conditions(graph, dataset.label_ntype, dataset.is_multi_label)
