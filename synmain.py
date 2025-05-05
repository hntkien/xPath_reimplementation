import json 
import os 
import logging 

import torch 
# import numpy as np
# import pandas as pd

from hgn import SimpleHeteroHGN
from syndata import SyntheticHGBDataset
from synxpath import xPathExplainer
from utils import load_xpath, filter_test_nodes, load_ground_truth_causes
from fidelity import eval_fidelity

# ========== Configurations ========== # 
HGN_TYPE = 'simplehgn'
DATASET = 'syn_acm'
TARGET_NTYPE = '0' # syntehtic datasets define the target node type as '0'
N_LAYER = 3
REPEAT_ID = 1 # experiment id
GPU = 0

# Define Paths 
abs_path = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(abs_path, 'log')
os.makedirs(log_dir, exist_ok=True)
data_dir = os.path.join(abs_path, 'data', DATASET)
result_dir = os.path.join(abs_path, 'results', HGN_TYPE)
os.makedirs(result_dir, exist_ok=True)
ckpt_dir = os.path.join(abs_path, 'ckpt', DATASET)
os.makedirs(ckpt_dir, exist_ok=True)

# Define Hyperparameters
if DATASET == 'syn_acm':
	XPATH_BEAM = 5
	XPATH_SAMPLE_N = 5
	XPATH_TOP_K = 4
# elif DATASET == 'syn_dblp' and HGN_TYPE == 'simplehgn' and N_LAYER == 2:
#     XPATH_BEAM = 10
#     XPATH_SAMPLE_N = 10
#     XPATH_TOP_K = 3
else:
	XPATH_BEAM = 2
	XPATH_SAMPLE_N = 10
	XPATH_TOP_K = 4

pred_list_path = f"{ckpt_dir}/{HGN_TYPE}_{DATASET}_pred_list_{N_LAYER}.json"
# To use trained model:
bk_dir = f"{ckpt_dir}/bk"
os.makedirs(bk_dir, exist_ok=True)
hgn_path = f"{bk_dir}/{HGN_TYPE}_{DATASET}_{N_LAYER}"
graph_path = f"{data_dir}/{DATASET}_processed_graph.bin"
result_path = f"{result_dir}/{DATASET}_l{N_LAYER}_xpath2s_{XPATH_BEAM}_{XPATH_SAMPLE_N}_exp{REPEAT_ID}"
cause_node_dict_path = os.path.join(ckpt_dir, f'cause_node_dict_{DATASET}_l{N_LAYER}_exp{REPEAT_ID}.json')

# Init logger
log_root = log_dir + f'/{HGN_TYPE}'
os.makedirs(log_root, exist_ok=True)
log_file = log_root + f'/{DATASET}_l{N_LAYER}_r{REPEAT_ID}_explain.log'
file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler()
fmt = '%(asctime)s - %(funcName)s - %(lineno)s - %(levelname)s - %(message)s'
formatter = logging.Formatter(fmt)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
logger = logging.getLogger('updateSecurity')
logger.setLevel('DEBUG')
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ========== End of Configurations ========== #

# ========== Helper Functions ========== #


# ========== Main Execution ========== #
if __name__ == "__main__": 
	device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

	logger.info(f'hgn_model: {HGN_TYPE}, dataset: {DATASET},'
				f'num_layer: {N_LAYER}, repeat_id: {REPEAT_ID}, device: {device}')

	# Load the model and graph
	dataset = SyntheticHGBDataset(dataset_name=DATASET, force_reload=True)
	graph = dataset[0]
	label_ntype = dataset.label_ntype
	labels = graph.nodes[label_ntype].data["label"]
	# graph = graph.to(device)
	is_multi_label = dataset.is_multi_label

	# # Test Nodes and Labels
	# test_nodes = graph.nodes[TARGET_NTYPE].data['test_mask'].nonzero(as_tuple=True)[0]
	# test_labels = graph.nodes[TARGET_NTYPE].data['label'][test_nodes]

	# if not is_multi_label:
	#     test_labels = test_labels.squeeze()

	# Load the ground truth causes
	_, target_nodes = load_ground_truth_causes(
		dataset_name=DATASET,
	)
	print(f"Number of target nodes: {len(target_nodes)}")
	target_nodes = torch.Tensor(target_nodes) 

	# ----- Load Model ----- #
	in_dim = {
		ntype: graph.nodes[ntype].data['feat'].shape[1] 
		for ntype in graph.ntypes
	}
	model = SimpleHeteroHGN(
		edge_dim=32,
		num_etypes=len(graph.etypes),
		in_dims=in_dim,
		num_hidden=32,
		num_classes=dataset.num_classes,
		num_layers=2,
		heads=[8] * N_LAYER,
		feat_drop=0.5,
		attn_drop=0.5,
		negative_slope=0.05,
		residual=True,
		alpha=0.05,
		shared_weight=True,
		is_multi_label=dataset.is_multi_label,
	)
	ckpt = torch.load(hgn_path, map_location=device)
	model.load_state_dict(ckpt["model_state"])
	model = model.to(device)
	model.eval()

	# if os.path.exists(pred_list_path):
	# 	#only generate explanation for correctly predicted nodes
	# 	explain_node = torch.Tensor(filter_test_nodes(
	# 		node_list=test_nodes, 
	# 		label=labels.tolist(), 
	# 		pred_list_path=pred_list_path))
	# else:
	# 	explain_node = test_nodes

	if os.path.exists(pred_list_path):
	#only generate explanation for correctly predicted nodes
		explain_node = torch.Tensor(filter_test_nodes(
			node_list=target_nodes, 
			label=labels.tolist(), 
			pred_list_path=pred_list_path))
	else:
		explain_node = target_nodes
	logger.info(f'Generating explanation for {len(explain_node)} samples...')

	explainer = xPathExplainer(
		model=model, 
		graph=graph, 
		target_ntype=TARGET_NTYPE, 
		num_layers=N_LAYER, 
		pred_list_path=pred_list_path, 
		device=device)
	
	xpath2s, cause_node_dict = explainer.explain_beam(
		graph=graph, 
		node_list=explain_node, 
		beam_size=XPATH_BEAM, 
		sample_n=XPATH_SAMPLE_N)
	
	with open(cause_node_dict_path, 'w') as f:
		json.dump(cause_node_dict, f)

	with open(result_path, 'w') as f:
		json.dump(xpath2s, f)

	logger.info(
		f'Loading xpath explanations: k={XPATH_TOP_K}, {result_path}')
	x, average_ne = load_xpath(result_path, XPATH_TOP_K)
	logger.info(f'Average neighborhood size {average_ne:.3f}')
	logger.info('Evaluating fidelity...')

	# # only evaluate explanation for correctly predicted nodes
	# explain_node = torch.Tensor(filter_test_nodes(
	# 		node_list=test_nodes, 
	# 		label=labels.tolist(), 
	# 		pred_list_path=pred_list_path))

	# only evaluate explanation for correctly predicted nodes
	explain_node = torch.Tensor(filter_test_nodes(
			node_list=target_nodes, 
			label=labels.tolist(), 
			pred_list_path=pred_list_path))
	
	facc, fprob = eval_fidelity(x, graph, model, labels.tolist(), TARGET_NTYPE, N_LAYER, dataset.num_classes, explain_node, device)
	logger.info(f'fmask acc:{facc:.5f}, fmask prob:{fprob:.5f}')

	