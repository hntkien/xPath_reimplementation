# %% 
import dgl 

dataset, label = dgl.load_graphs('/home/kirin/kirin_master/xPath_reimplementation/data/acm/hgb_acm.bin')
graph = dataset[0]
print(graph)
print(label)
# %%
import torch 
# Get number of nodes for each node type
for ntype in graph.ntypes:
    print(f"Number of {ntype} nodes:", graph.number_of_nodes(ntype))

# Get number of edges for each edge type
for etype in graph.canonical_etypes:
    print(f"Number of {etype} edges:", graph.number_of_edges(etype))

# Get labels if available
print(graph.ndata)
target_ntype = list(graph.ndata['label'].keys())[0]
print(f"Target node type for labels: {target_ntype}")
print("Labels:", torch.unique(graph.ndata['label'][target_ntype]))

# Get masks
for mask in ['train_mask', 'val_mask', 'test_mask']:
    if mask in graph.ndata:
        print(f"Number of {mask.split('_')[0]} samples:", graph.nodes[target_ntype].data[mask].sum().item())
        
# %%
train_nodes = graph.nodes[target_ntype].data['train_mask'].nonzero().squeeze()
print(train_nodes)
# %%
for ntype in graph[0].ntypes:
    features = graph[0].nodes[ntype].data['feat']
    print(f"Feature shape for {ntype} nodes:", features.shape)
# %%
_info = torch.load('/home/kirin/kirin_master/xPath_reimplementation/data/acm/acm_index_60.bin')
print(_info.keys())
print(_info['train_index'])
print(_info['valid_index'])
print(_info['test_index'])
print(_info['test_label'])
# %%
