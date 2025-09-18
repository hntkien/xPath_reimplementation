import json
import torch
import os

from utils import get_model, filter_test_nodes, load_xpath
from xpath import xPath_Explainer
from fidelity import eval_fidelity
from config import GPU, HGN_TYPE, DATASET, TARGET_NTYPE, N_LAYER, NUM_CLASSES, REPEAT_ID, XPATH_BEAM, XPATH_SAMPLE_N, XPATH_TOP_K
from config import graph_path, index_path, hgn_path, pred_list_path, result_path, logger


if __name__ == '__main__':
    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

    logger.info(f'hgn_model: {HGN_TYPE}, dataset: {DATASET},'
                f'num_layer: {N_LAYER}, repeat_id: {REPEAT_ID}, device: {device}')

    g, model = get_model(HGN_TYPE, N_LAYER, graph_path)
    ckpt = torch.load(hgn_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval() 
    # ---------- Done ----------

    # ----- Extract Labels and Indices -----
    target_ntype = list(g.ndata['label'].keys())[0]
    labels = g.nodes[target_ntype].data["label"] 
    num_classes = torch.unique(labels).shape[0]
    test_masks = g.nodes[target_ntype].data["test_mask"].to(torch.bool)
    # Extract indices 
    test_nodes = test_masks.nonzero().squeeze()
    # Extract labels
    test_labels = labels[test_masks]

    if os.path.exists(pred_list_path):
        #only generate explanation for correctly predicted nodes
        explain_node = torch.Tensor(filter_test_nodes(test_nodes, labels, pred_list_path))
    else:
        explain_node = torch.Tensor(test_nodes)
    logger.info(f'Generating explanation for {len(explain_node)} samples...')

    explainer = xPath_Explainer(model, g, target_ntype, N_LAYER, pred_list_path, device)
    xpath2s = explainer.explain_beam(g, explain_node, beam=XPATH_BEAM, sample_n=XPATH_SAMPLE_N)
    with open(result_path, 'w') as f:
        json.dump(xpath2s, f)

    logger.info(f'Loading xpath explanations: k={XPATH_TOP_K}, {result_path}')
    x, average_ne = load_xpath(result_path, XPATH_TOP_K)
    logger.info(f'Average neighborhood size {average_ne:.3f}')
    logger.info('Evaluating fidelity...')
    # only evaluate explanation for correctly predicted nodes
    explain_node = torch.Tensor(filter_test_nodes(test_nodes, labels, pred_list_path))
    facc, fprob = eval_fidelity(x, g, model, labels.tolist(), target_ntype, N_LAYER, num_classes, explain_node, device)
    logger.info(f'fmask acc:{facc:.5f}, fmask prob:{fprob:.5f}')
