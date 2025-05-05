import json 
import random 
import numpy as np 
import torch 
import torch.nn.functional as F 
import dgl 
from tqdm import tqdm 

# ========== Helper Functions ========== #
def get_original_name(path_str: str, origin_ids: dict) -> str:
	"""
	Converts internal node IDs in a path string to their original IDs using a mapping dictionary.
	
	This function processes a path string of the form "type1-id1-type2-id2-..." where each id is an internal/local ID, and converts it to use the original/global node IDs.
	
	Args:
		path_str (str): A string representing a path in the format 
			"type1-id1-type2-id2-..." ending with a hyphen.
		origin_ids (dict): A dictionary mapping node types to lists of original 
			IDs where the index is the internal ID. Format: {node_type: [original_id1, original_id2, ...]}
	
	Returns:
		A string representing the path with original node IDs in the format "type1-orig_id1,type2-orig_id2,...".
	
	Example:
		If path_str = "paper-0-author-1-" and origin_ids = {"paper": [42, 43], "author": [101, 102, 103]},
		then the result would be "paper-42,author-102,"
	"""
	# Remove trailing hyphen and split by hyphen
	path_elements = path_str[:-1].split('-')
	
	i = 0
	original_path = ""
	
	# Process pairs of (node_type, node_id)
	while i < len(path_elements):
		node_type = path_elements[i]
		internal_id = int(path_elements[i + 1])
		
		# Map internal ID to original ID using the origin_ids dictionary
		original_id = origin_ids[node_type][internal_id]
		original_path += f"{node_type}-{original_id},"
		
		i += 2
		
	return original_path

def get_score(
		original_probs: list, 
		modified_probs: list, 
		label: int = 0) -> float:
	"""
	Calculates an influence score that measures how a model's prediction has changed.
	
	This function quantifies the impact of a modification on model predictions by comparing probability distributions. It penalizes changes that flip the prediction and rewards those that strengthen the confidence in the original label.
	
	Args:
		original_probs (list): List of probabilities from the original model prediction
		modified_probs (list): List of probabilities from the model after some modification
		label: The index of the class of interest (default: 0)
	
	Returns:
		float: The influence score. 
	
	"""
	# Find the class with highest probability in the modified prediction
	predicted_class = np.argmax(modified_probs)
	
	# Determine base score based on whether prediction changed
	if label == predicted_class:
		base_score = -1  # Prediction stayed the same
	else:
		base_score = 1   # Prediction changed
	
	# Add the difference in probability for the class of interest
	# (measures how much the confidence changed)
	probability_diff = original_probs[label] - modified_probs[label]
	
	return base_score + probability_diff

# ========== Class Definitions ========== #
class xPathExplainer:
	def __init__(
			self, 
			model, 
			graph, 
			target_ntype, 
			num_layers, 
			pred_list_path, 
			device,
	):
		self.target_ntype = target_ntype 
		self.model = model 
		self.device = device 
		self.model.eval() 
		self.num_layers = num_layers 
		self.pred_list_path = pred_list_path 
		self.n2etp = {} 

		for src_ntype, etype, dst_ntype in graph.canonical_etypes: 
			self.n2etp[(src_ntype, dst_ntype)] = (src_ntype, etype, dst_ntype) 

		self.prediction_list = {} 
		self.one_hop_sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1) 

	def sample_step(self, graph, ap, sample_n) -> dict:
		"""
		Samples a subgraph based on the given path and node ID.
		
		Args:
			graph (DGLGraph): The input graph.
			ap (str): The path string.
			sample_n (int): The number of nodes to sample.
		
		Returns:
			dict: A dictionary where keys are node types and values are lists of sampled node IDs.
		"""
		# Split the path string into node types and IDs
		tmp = ap[:-1].split('-')
		aptp = [tmp[i] for i in range(0, len(tmp), 2)]
		apid = [int(tmp[i]) for i in range(1, len(tmp), 2)]
		res = {}
		
		# Create a NodeDataLoader for one-hop sampling
		one_hop_loader = dgl.dataloading.DataLoader(
			graph=graph, 
			indices={
				aptp[-1]: torch.tensor([apid[-1]], dtype=torch.int64, device=self.device)}, 
			graph_sampler=self.one_hop_sampler, 
			batch_size=1, 
			shuffle=False, 
			drop_last=False) 
		
		for neighbors, _, _ in one_hop_loader:
			for tp in neighbors:
				res[tp] = neighbors[tp].detach().cpu().tolist()
				if len(res[tp]) > sample_n:
					res[tp] = random.sample(res[tp], sample_n)
			for tp, aid in zip(aptp, apid):
				if aid in res[tp]:
					res[tp].remove(aid)
			break

		path = {}
		for tp in res:
			if len(res[tp]) > 0:
				mptp = tuple(aptp + [tp])
				path[mptp] = [apid + [i] for i in res[tp]]
		
		return path
	
	def get_proxy_graph(self, graph, mptp, p):
		"""_summary_

		Args:
			graph (_type_): _description_
			mptp (_type_): _description_
			p (_type_): _description_
		"""
		graph = graph.to("cpu") 

		if len(p) == 1:
			return 
		
		x = {
			tp: graph.nodes[tp].data['feat'].clone().detach().cpu() 
			for tp in graph.ntypes}
		sg_n = {n: graph.nodes[n].data["feat"].shape[0] for n in graph.ntypes}
		proxy_ids = [-1]

		for i in range(1, len(mptp) - 1):
			node_tp = mptp[i]
			node_id = p[i]
			node_feature = x[node_tp][node_id, :].reshape(shape=(1, -1))

			proxy_ids.append(sg_n[node_tp])
			x[node_tp] = torch.concat([x[node_tp], node_feature], dim=0)
			sg_n[node_tp] += 1

		sg_edges = {}

		for stp, etp, ttp in graph.canonical_etypes:
			sg_edges[(stp, etp, ttp)] = [
				graph.edges(etype=etp)[0].tolist(), 
				graph.edges(etype=etp)[1].tolist()
			]
		new_edges = {k: [[], []] for k in graph.canonical_etypes}

		for i in range(1, len(mptp)):
			node_tp = mptp[i]

			if i != len(mptp) - 1:
				if (node_tp, node_tp) in self.n2etp:
					etp = self.n2etp[(node_tp, node_tp)]
					new_edges[etp][0] += [proxy_ids[i]]
					new_edges[etp][1] += [proxy_ids[i]]
				else:
					# Skip this connection or log a message
					print(f"Warning: No edge type defined between {node_tp} and itself")

				etp = self.n2etp[(node_tp, mptp[i + 1])]
				if i != len(mptp) - 2:
					new_edges[etp][0] += [proxy_ids[i]]
					new_edges[etp][1] += [proxy_ids[i + 1]]
				else:
					new_edges[etp][0] += [proxy_ids[i]]
					new_edges[etp][1] += [p[-1]]

			if i != 1:
				etp = self.n2etp[(node_tp, mptp[i - 1])]
				if i != len(mptp) - 1:
					new_edges[etp][0] += [proxy_ids[i]]
					new_edges[etp][1] += [proxy_ids[i - 1]]
				else:
					new_edges[etp][0] += [p[-1]]
					new_edges[etp][1] += [proxy_ids[i - 1]]
		
		del_id = -1
		for stp, etp, dtp in graph.canonical_etypes:
			for i in range(1, len(mptp) - 1):
				if stp == mptp[i]:
					for j in range(len(sg_edges[(stp, etp, dtp)][0])):
						sid = sg_edges[(stp, etp, dtp)][0][j]
						tid = sg_edges[(stp, etp, dtp)][1][j]
						if sid == p[i]:
							if (dtp != mptp[i - 1] or tid != p[i - 1]) \
									and (dtp != mptp[i] or tid != p[i]) \
									and (dtp != mptp[i + 1] or tid != p[i + 1]):
								new_edges[(stp, etp, dtp)][0] += [proxy_ids[i]]
								new_edges[(stp, etp, dtp)][1] += [tid]
			if stp == mptp[-1] and dtp == mptp[-2]:
				for j in range(len(sg_edges[(stp, etp, dtp)][0])):
					if sg_edges[(stp, etp, dtp)][0][j] == p[-1] and sg_edges[(stp, etp, dtp)][1][j] == p[-2]:
						del_id = j
						break
				sg_edges[(stp, etp, dtp)][0].pop(del_id)
				sg_edges[(stp, etp, dtp)][1].pop(del_id)

			sg_edges[(stp, etp, dtp)][0] += new_edges[((stp, etp, dtp))][0]
			sg_edges[(stp, etp, dtp)][1] += new_edges[((stp, etp, dtp))][1]
			sg_edges[(stp, etp, dtp)] = (
				sg_edges[(stp, etp, dtp)][0], sg_edges[(stp, etp, dtp)][1])
			
		sg = dgl.heterograph(sg_edges)
		for tp in graph.ntypes:
			sg.nodes[tp].data['nfeat'] = x[tp]

		return sg 
	
	def explain_beam(
			self, 
			graph, 
			node_list, 
			beam_size=3, 
			sample_n=10,
	):
		"""
		Generates explanations for a given node ID using beam search.
		
		Args:
			graph (DGLGraph): The input graph.
			node_list (int): The list of the nodes to explain.
			beam_size (int): The number of paths to keep at each step of the beam search.
			sample_n (int): The number of nodes to sample in each step.
		
		Returns:
			Returns:
				tuple: 
				- xpath (dict): Maps each target node ID to a dict of path string → importance score.
				- cause_nodes_dict (dict): Maps each target node ID to a list of (node_type, node_id) tuples that contributed to prediction.
		"""
		sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.num_layers)
		subgraph_dataloader = dgl.dataloading.DataLoader(
			graph=graph, 
			indices={self.target_ntype: node_list.type(torch.int64)}, 
			graph_sampler=sampler, 
			batch_size=1, 
			shuffle=False, 
			drop_last=False)
		
		j = 0 
		xpath = {} 
		cause_nodes_dict = {} 

		for neighbors, _, _ in tqdm(subgraph_dataloader):
			subgraph = dgl.node_subgraph(graph, neighbors)
			subgraph = subgraph.to(self.device) 
			target = node_list[j] 

			# Get the original node ID
			original_id = {
				tp: subgraph.nodes[tp].data["_ID"].tolist() 
				for tp in subgraph.ntypes}
			# id_target = original_id[self.target_ntype].index(target)
			id_target = subgraph.nodes[
				self.target_ntype].data[dgl.NID].tolist().index(target)

			x = {
				tp: subgraph.nodes[tp].data["feat"].clone() 
				for tp in subgraph.ntypes}
			
			self.model.g = subgraph 
			logits = self.model.forward(x, self.target_ntype)

			origin_probs = F.softmax(
				logits[id_target], dim=-1).detach().cpu().tolist() 
			origin_label = np.argmax(origin_probs)
			self.prediction_list[int(target.item())] = origin_label.item() 

			ancestor_paths = [f"{self.target_ntype}-{id_target}-"]
			top_k_paths = {f"{self.target_ntype}-{id_target}-": -100}
			visited = {}
			path_scores = {}

			while len(ancestor_paths) > 0:
				for ancestor_path in ancestor_paths:
					paths = self.sample_step(
						subgraph, ancestor_path, sample_n=sample_n)
					
					for mptp in paths:
						for pid in range(len(paths[mptp])):
							path = paths[mptp][pid]
							path_key = ""
							
							for k, tp in enumerate(mptp):
								path_key += f"{tp}-{path[k]}-"
							
							shadow_graph = self.get_proxy_graph(
								subgraph, mptp, path).to(self.device)
							x = shadow_graph.ndata.pop("nfeat")
							self.model.g = shadow_graph 
							logits = self.model.forward(x, self.target_ntype)
							y = logits[id_target]
							probabilities = F.softmax(y, dim=-1).detach().cpu().tolist()
							
							path_scores[path_key] = get_score(
								origin_probs, probabilities, label=origin_label)
							top_k_paths[path_key] = path_scores[path_key]
				
				values = list(top_k_paths.values())
				keys = list(top_k_paths.keys())
				indices = np.argsort(values)[-beam_size:]
				top_k_paths = {keys[b]: top_k_paths[keys[b]] for b in indices}

				ancestor_paths = []
				for b in top_k_paths:
					tmp = b[:-1].split('-')
					if (len(tmp) <= 2 * self.num_layers) and (not b in visited):
						ancestor_paths.append(b)
						visited[b] = 1

			path_scores = {
				get_original_name(i, original_id): path_scores[i] 
				for i in path_scores}
			target = int(target.item())
			xpath[target] = path_scores

			cause_nodes = set()

			for path_str in path_scores:
				p = path_str[:-1].split(',')
				for n in p:
					if not n: 
						continue 
					tp, nid = n.split('-')
					if int(nid) != target:
						cause_nodes.add((tp, int(nid)))
						
				# for i in range(0, len(p), 2):
				# 	tp, nid = p[i], p[i+1]
				# 	if int(nid) != target:
				# 		cause_nodes.add((tp, int(nid)))

			cause_nodes_dict[target] = sorted(list(cause_nodes))

			j += 1
		
		with open(self.pred_list_path, 'w') as f:
			json.dump(self.prediction_list, f)

		return xpath, cause_nodes_dict

