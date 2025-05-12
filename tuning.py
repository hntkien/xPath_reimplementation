import copy 

import torch 
import numpy as np 
from itertools import product 

from hgn import SimpleHeteroHGN
from syndata import SyntheticHGBDataset 
from synconfig import PATHS, DATASET_CONFIG, MODEL_CONFIG, logger 
from sklearn.metrics import f1_score

# ========== F1 Score Function ========== #
def f1_scores(logits, labels):
    """
    Calculate micro and macro F1 scores for multi-class classification.

    Args:
        logits (torch.Tensor): Model output logits.
        labels (torch.Tensor): Ground truth labels.

    Returns:
        tuple: Micro F1 score, Macro F1 score.
    """
    pred = logits.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    micro_f1 = f1_score(labels, pred, average='micro')
    macro_f1 = f1_score(labels, pred, average='macro')
    return micro_f1, macro_f1


# ========== Main Execution ========== # 
if __name__ == "__main__":
    device = torch.device(
        f"cuda:{MODEL_CONFIG['gpu']}" if torch.cuda.is_available() else "cpu")

    # ----- Load Dataset ----- #
    dataset = SyntheticHGBDataset(
        dataset_name=DATASET_CONFIG["dataset_name"], 
        force_reload=DATASET_CONFIG["force_reload"])
    graph = dataset[0]
    # is_multi_label = dataset.is_multi_label 

    # Temporarily set is_multi_label to False for testing
    is_multi_label = False 
    target_ntype = DATASET_CONFIG["target_ntype"]

    # Train Nodes and Labels 
    train_node = graph.nodes[target_ntype].data['train_mask'].nonzero(as_tuple=True)[0]
    train_label = graph.nodes[target_ntype].data['label'][train_node]

    # Validation Nodes and Labels 
    valid_node = graph.nodes[target_ntype].data['val_mask'].nonzero(as_tuple=True)[0]
    valid_label = graph.nodes[target_ntype].data['label'][valid_node]

    # Test Nodes and Labels
    test_node = graph.nodes[target_ntype].data['test_mask'].nonzero(as_tuple=True)[0]
    test_label = graph.nodes[target_ntype].data['label'][test_node]

    # If using multi-label, labels are likely in one-hot or binary multi-label format
    # Otherwise, they are integers (class indices)
    if not is_multi_label:
        train_label = train_label.squeeze()
        valid_label = valid_label.squeeze()
        test_label = test_label.squeeze()

    # Define hyperparameter grid 
    param_grid = {
        "edge_dim": [16, 32, 64], 
        "num_hidden": [16, 32, 64],
        "num_layers": [2, 3],
        "lr": [0.0001, 0.001, 0.01],
    }
    # Generate all combinations of hyperparameters
    param_combinations = list(product(
        param_grid["edge_dim"],
        param_grid["num_hidden"],
        param_grid["num_layers"],
        param_grid["lr"],
    ))
    
    # ----- Training ----- #
    best_score = 0
    max_score = 0

    # Data 
    in_dim = {
        ntype: graph.nodes[ntype].data['feat'].shape[1] 
        for ntype in graph.ntypes
    }
    x = {
        ntype: graph.nodes[ntype].data['feat'].to(device) 
        for ntype in graph.ntypes
    }
    train_node, train_label = train_node.to(device), train_label.to(device)
    valid_node, valid_label = valid_node.to(device), valid_label.to(device)
    test_node, test_label = test_node.to(device), test_label.to(device)

    # ----- Load Model ----- #
    for params in param_combinations:
        edge_dim, num_hidden, num_layers, lr = params
        logger.info(f"Training with params: {params}")

        model = SimpleHeteroHGN(
            edge_dim=edge_dim,
            num_etypes=len(graph.etypes),
            in_dims=in_dim,
            num_hidden=num_hidden,
            num_classes=dataset.num_classes,
            num_layers=num_layers,
            heads=[8] * num_layers,
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
            lr=lr, 
            weight_decay=5e-5)

        # Training loop
        patience = 0
        min_loss = np.inf
        epoch = 0
        log_epoch = 100
        max_epoch = 2000
        max_patience = 50

        for epoch in range(max_epoch):
            model.train()
            optimizer.zero_grad()
            loss = model.loss(x, target_ntype, train_node, train_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

            if epoch % log_epoch == 0:
                logger.info(f"Epoch: {epoch} | Loss: {loss.item():.4f}")

            # Validation
            model.eval()
            with torch.no_grad():
                logits = model.forward(x, target_ntype)
                train_logits = logits[train_node]
                valid_logits = logits[valid_node]

                train_loss = model.cross_entropy_loss(train_logits, train_label).item()
                val_loss = model.cross_entropy_loss(valid_logits, valid_label).item()
                train_micro, train_macro = f1_scores(train_logits, train_label)
                val_micro, val_macro = f1_scores(valid_logits, valid_label)

            # Early stopping
            if val_loss <= min_loss or val_macro >= max_score:
                if val_macro >= best_score:
                    best_score = val_macro
                    # best_model = copy.deepcopy(model.state_dict())
                    best_model = model
                    best_params = params
                min_loss = min(min_loss, val_loss)
                max_score = max(max_score, val_macro)
                patience = 0
            else:
                patience += 1
                if patience == max_patience:
                    logger.info("Early stopping triggered.")
                    # model.load_state_dict(best_model)
                    break

    # model.load_state_dict(best_model)
    logger.info(f"Best Hyperparameters: {best_params}")
    model = best_model 

    # Test
    model.eval()
    with torch.no_grad():
        logits = model.forward(x, target_ntype)
        test_logits = logits[test_node]
        test_micro, test_macro = f1_scores(test_logits, test_label)
        logger.info(f"Test Micro-F1: {test_micro:.5f} \t Test Macro-F1: {test_macro:.5f}")

    # Save model
    torch.save(
        {
            "epoch": epoch,
            "model_type": "SimpleHeteroHGN",
            "optimizer": optimizer,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, 
        PATHS["hgn_path"])