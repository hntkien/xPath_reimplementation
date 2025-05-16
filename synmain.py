import json 
import os 

import torch 

from hgn import SimpleHeteroHGN
from syndata import SyntheticHGBDataset
from synxpath import xPathExplainer
from utils import load_xpath, filter_test_nodes, load_ground_truth_causes
from fidelity import eval_fidelity
from synconfig import logger, PATHS, DATASET_CONFIG, MODEL_CONFIG, EXPLAIN_CONFIG, REPEAT_ID 

# if DATASET_CONFIG["dataset_name"] == "syn_dblp":
# 	edge_dim = 16
# 	num_hidden = 32
# 	n_layers = 2 

# ========== Main Execution ========== #
if __name__ == "__main__": 
	device = torch.device(
		f"cuda:{MODEL_CONFIG['gpu']}" if torch.cuda.is_available() else "cpu")

	logger.info(
		f'hgn_model: {MODEL_CONFIG["hgn_type"]}, dataset: {DATASET_CONFIG["dataset_name"]},'
		f'num_layer: {MODEL_CONFIG["n_layer"]}, repeat_id: {REPEAT_ID}, device: {device}')

	# Load the model and graph
	dataset = SyntheticHGBDataset(
		dataset_name=DATASET_CONFIG["dataset_name"],
		force_reload=DATASET_CONFIG["force_reload"])
	graph = dataset[0]
	label_ntype = dataset.label_ntype
	labels = graph.nodes[label_ntype].data["label"]
	# graph = graph.to(device)

	# is_multi_label = dataset.is_multi_label
	is_multi_label = False

	# # Test Nodes and Labels
	# test_nodes = graph.nodes[TARGET_NTYPE].data['test_mask'].nonzero(as_tuple=True)[0]
	# test_labels = graph.nodes[TARGET_NTYPE].data['label'][test_nodes]

	# if not is_multi_label:
	#     test_labels = test_labels.squeeze()

	# Load the ground truth causes
	_, target_nodes = load_ground_truth_causes(
		dataset_name=DATASET_CONFIG["dataset_name"],
	)
	print(f"Number of target nodes: {len(target_nodes)}")
	target_nodes = torch.Tensor(target_nodes) 

	# ----- Load Model ----- #
	in_dim = {
		ntype: graph.nodes[ntype].data['feat'].shape[1] 
		for ntype in graph.ntypes
	}
	model = SimpleHeteroHGN(
		edge_dim=64,
		num_etypes=len(graph.etypes),
		in_dims=in_dim,
		num_hidden=64,
		num_classes=dataset.num_classes,
		num_layers=MODEL_CONFIG["n_layer"],
		heads=[8] * MODEL_CONFIG["n_layer"],
		feat_drop=0.5,
		attn_drop=0.5,
		negative_slope=MODEL_CONFIG["neg_slope"],
		# negative_slope=0.05,
		residual=True,
		alpha=0.05,
		shared_weight=True,
		is_multi_label=is_multi_label,
	)
	ckpt = torch.load(PATHS["hgn_path"], map_location=device)
	model.load_state_dict(ckpt["model_state"])
	model = model.to(device)
	model.eval()

	pred_list_path = PATHS["pred_list_path"]

	if os.path.exists(pred_list_path):
		# only generate explanation for correctly predicted nodes
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
		target_ntype=DATASET_CONFIG["target_ntype"], 
		num_layers=MODEL_CONFIG["n_layer"], 
		pred_list_path=pred_list_path, 
		device=device)
	
	xpath2s, cause_node_dict = explainer.explain_beam(
		graph=graph, 
		node_list=explain_node, 
		beam_size=EXPLAIN_CONFIG["xpath_beam"], 
		sample_n=EXPLAIN_CONFIG["xpath_sample_n"],)
	
	with open(PATHS["cause_node_dict_path"], 'w') as f:
		json.dump(cause_node_dict, f)

	with open(PATHS["result_path"], 'w') as f:
		json.dump(xpath2s, f)

	logger.info(
		f'Loading xpath explanations: k={EXPLAIN_CONFIG["xpath_top_k"]}, {PATHS["result_path"]}')
	x, average_ne = load_xpath(PATHS["result_path"], EXPLAIN_CONFIG["xpath_top_k"])
	logger.info(f'Average neighborhood size {average_ne:.3f}')
	logger.info('Evaluating fidelity...')

	# only evaluate explanation for correctly predicted nodes
	explain_node = torch.Tensor(filter_test_nodes(
			node_list=target_nodes, 
			label=labels.tolist(), 
			pred_list_path=pred_list_path))
	
	facc, fprob = eval_fidelity(x, graph, model, labels.tolist(), DATASET_CONFIG["target_ntype"], MODEL_CONFIG["n_layer"], dataset.num_classes, explain_node, device)
	logger.info(f'fmask acc:{facc*100:.3f}, fmask prob:{fprob*100:.3f}')

	