import os 
import logging 
import copy 

import torch 
import numpy as np 

from hgn import SimpleHeteroHGN
# from utils import accuracy 
from syndata import SyntheticHGBDataset 

# ========== Configurations ========== # 
HGN_TYPE = 'simplehgn'
DATASET = 'syn_recipe'
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
result_path = f'{result_dir}/{DATASET}_l{N_LAYER}_xpath2s_{XPATH_BEAM}_{XPATH_SAMPLE_N}_exp{REPEAT_ID}'

# Init logger
log_root = log_dir + f'/{HGN_TYPE}'
os.makedirs(log_root, exist_ok=True)
log_file = log_root + f'/{DATASET}_l{N_LAYER}_r{REPEAT_ID}.log'
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

# ========== Accuracy Function ========== #
def accuracy(logits, labels):
    if is_multi_label:
        pred = (logits > 0).float()
        correct = (pred == labels).float().mean().item()
        return correct
    else:
        pred = logits.argmax(dim=1)
        correct = (pred == labels).float().mean().item()
        return correct


# ========== Main Execution ========== # 
if __name__ == "__main__":
    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")

    # ----- Load Dataset ----- #
    dataset = SyntheticHGBDataset(dataset_name=DATASET, force_reload=True)
    graph = dataset[0]
    # is_multi_label = dataset.is_multi_label 
    # Temporarily set is_multi_label to False for testing
    is_multi_label = False 

    # Train Nodes and Labels 
    train_node = graph.nodes[TARGET_NTYPE].data['train_mask'].nonzero(as_tuple=True)[0]
    train_label = graph.nodes[TARGET_NTYPE].data['label'][train_node]

    # Validation Nodes and Labels 
    valid_node = graph.nodes[TARGET_NTYPE].data['val_mask'].nonzero(as_tuple=True)[0]
    valid_label = graph.nodes[TARGET_NTYPE].data['label'][valid_node]

    # Test Nodes and Labels
    test_node = graph.nodes[TARGET_NTYPE].data['test_mask'].nonzero(as_tuple=True)[0]
    test_label = graph.nodes[TARGET_NTYPE].data['label'][test_node]

    # If using multi-label, labels are likely in one-hot or binary multi-label format
    # Otherwise, they are integers (class indices)
    if not is_multi_label:
        train_label = train_label.squeeze()
        valid_label = valid_label.squeeze()
        test_label = test_label.squeeze()

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
        is_multi_label=is_multi_label,
    )
    model.to(device) 
    model.g = graph.to(device) 
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.0001, 
        weight_decay=5e-5)
    
    # ----- Training ----- #
    patience = 0
    best_score = 0
    max_score = 0
    min_loss = np.inf
    epoch = 0
    log_epoch = 100
    max_epoch = 2000
    max_patience = 50

    # Data 
    x = {
        ntype: graph.nodes[ntype].data['feat'].to(device) 
        for ntype in graph.ntypes
    }
    train_node, train_label = train_node.to(device), train_label.to(device)
    valid_node, valid_label = valid_node.to(device), valid_label.to(device)
    test_node, test_label = test_node.to(device), test_label.to(device)

    for epoch in range(max_epoch):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(x, TARGET_NTYPE, train_node, train_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()

        if epoch % log_epoch == 0:
            logger.info(f"Epoch: {epoch} | Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model.forward(x, TARGET_NTYPE)
            train_logits = logits[train_node]
            valid_logits = logits[valid_node]

            train_loss = model.cross_entropy_loss(train_logits, train_label).item()
            val_loss = model.cross_entropy_loss(valid_logits, valid_label).item()
            train_acc = accuracy(train_logits, train_label)
            val_acc = accuracy(valid_logits, valid_label)

        if epoch % log_epoch == 0:
            logger.info(
                f"Train ACC: {train_acc:.3f} | Train Loss: {train_loss:.3f} | "
                f"Val ACC: {val_acc:.3f} | Val Loss: {val_loss:.3f}")

        # Early stopping
        if val_loss <= min_loss or val_acc >= max_score:
            if val_acc >= best_score:
                best_score = val_acc
                best_model = copy.deepcopy(model.state_dict())
            min_loss = min(min_loss, val_loss)
            max_score = max(max_score, val_acc)
            patience = 0
        else:
            patience += 1
            if patience == max_patience:
                logger.info("Early stopping triggered.")
                model.load_state_dict(best_model)
                break

    # Test
    model.eval()
    with torch.no_grad():
        logits = model.forward(x, TARGET_NTYPE)
        test_logits = logits[test_node]
        test_acc = accuracy(test_logits, test_label)
        logger.info(f"Test ACC: {test_acc:.3f}")

    # Save model
    torch.save({
        "epoch": epoch,
        "model_type": "SimpleHeteroHGN",
        "optimizer": optimizer,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, hgn_path)