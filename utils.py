import os
import re
import json
import torch
import dgl
import hgn
import numpy as np
import pandas as pd 

def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)


def get_model(hgn_model_type, n_layer, num_classes, graph_path, index_path):
    model, _info = None, None
    gs, _ = dgl.load_graphs(graph_path)
    g = gs[0]
    _info = torch.load(index_path)
    if hgn_model_type == 'simplehgn':
        in_dim = {n: g.nodes[n].data['nfeat'].shape[1] for n in g.ntypes}
        edge_type_num = len(g.etypes)

        model = hgn.SimpleHeteroHGN(32, edge_type_num, in_dim, 32, num_classes, 
                                    n_layer, [8] * n_layer, 0.5, 0.5, 0.05, True, 0.05, shared_weight=True)

    elif hgn_model_type == 'hgt':
        node_dict = {}
        edge_dict = {}
        n_inp = {}
        for ntype in g.ntypes:
            node_dict[ntype] = len(node_dict)
            n_inp[node_dict[ntype]] = g.nodes[ntype].data['nfeat'].shape[1]
        for etype in g.etypes:
            edge_dict[etype] = len(edge_dict)
            g.edges[etype].data['id'] = torch.ones(g.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]
        model = hgn.HGT(node_dict, edge_dict, n_inp=n_inp, n_hid=32, 
                        n_out=num_classes, n_layers=n_layer, n_heads=4, 
                        use_norm=True)

    return g, model, _info


def filter_test_nodes(node_list, label, pred_list_path):
    with open(pred_list_path, 'r') as f:
        pred_list = json.load(f)

    correct_node_list = []
    for i in range(len(node_list)):
        target_id = int(node_list[i].item())
        if str(target_id) not in pred_list:
            continue
        if label[target_id] == pred_list[str(target_id)]:
            correct_node_list.append(target_id)
    return correct_node_list


def load_xpath(result_path, k=5):
    c = 0
    best = True  # fine-grained explantions with best scores
    th = -1
    with open(result_path, 'r') as f:
        xpath = json.load(f)

    res = {}
    for x in xpath:
        paths = []
        s_values = list(xpath[x].values())
        path_names = list(xpath[x].keys())
        if best:
            ind = np.argsort(s_values)[-k:]
        else:
            ind = np.argsort(s_values)[:k]

        tmp_n = []
        for j, pid in enumerate(ind):
            path_nodes = path_names[pid][:-1].split(',')

            if j != len(ind) - 1 and s_values[pid] < th:
                continue

            tmp = []
            tmp_n += path_nodes
            for n in path_nodes:
                tmp.append((n.split('-')[0], int(n.split('-')[1])))
            paths.append(tmp)
        c += len(set(tmp_n))
        res[int(x)] = paths

    return res, c / len(res)

def load_ground_truth_causes(
        dataset_name: str, 
        base_path: str="./data") -> tuple:
    """Constructs a mapping from target node ID (from motif filename) to cause node IDs. 

    Args:
        dataset_name (str): Name of the dataset. 
        base_path (str, optional): Base path where datasets are stored. 
            Defaults to "./data".

    Returns:
        tuple: 
            - cause_dict (dict[str, list[int]]): target node ID -> list of cause node IDs. 
            - target_nodes (list[int]): All target node IDs found in the motif files. 
    """
    dataset_path = os.path.join(base_path, dataset_name, "motifs")
    motif_pattern = re.compile(r"motif_(\d+)\.csv")

    cause_dict = {} 
    target_nodes = [] 

    all_files = os.listdir(dataset_path)
    # print(f"Scanning {len(all_files)} files in: {dataset_path}")

    for filename in all_files:
        match = motif_pattern.match(filename)

        if match: 
            target_id = int(match.group(1)) 
            file_path = os.path.join(dataset_path, filename)
            # print(f"Processing {filename} as target node {target_id}")

            df = pd.read_csv(file_path, header=0, names=["src", "dst", "etype"])

            # Cause nodes are all nodes except the target node 
            # related_nodes = set(df["src"].tolist() + df["dst"].tolist())
            # cause_nodes = list(related_nodes - {target_id})

            related_nodes = set(df["src"].tolist())

            cause_dict[str(target_id)] = list(related_nodes)
            target_nodes.append(target_id)
        
    return cause_dict, target_nodes