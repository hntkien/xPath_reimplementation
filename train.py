import torch
import numpy as np
import copy
from config import GPU, TARGET_NTYPE, HGN_TYPE, N_LAYER, NUM_CLASSES, hgn_path, graph_path, index_path, logger
from utils import get_model, accuracy


if __name__ == '__main__':
    device = torch.device(f"cuda:{GPU}" if torch.cuda.is_available() else "cpu")
    g, model = get_model(HGN_TYPE, N_LAYER, graph_path)

    model.to(device)
    model.g = g.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5)
    target_ntype = list(g.ndata['label'].keys())[0]
    labels = g.nodes[target_ntype].data["label"] 
    train_masks = g.nodes[target_ntype].data["train_mask"].to(torch.bool)
    val_masks = g.nodes[target_ntype].data["val_mask"].to(torch.bool)
    test_masks = g.nodes[target_ntype].data["test_mask"].to(torch.bool)
    # Extract indices 
    train_nodes = train_masks.nonzero().squeeze().long().to(device)
    val_nodes = val_masks.nonzero().squeeze().long().to(device)
    test_nodes = test_masks.nonzero().squeeze().long().to(device)
    # Extract labels
    train_labels = labels[train_masks].long().to(device)
    val_labels = labels[val_masks].long().to(device)
    test_labels = labels[test_masks].long().to(device)

    patience = 0
    best_score = 0
    max_score = 0
    min_loss = np.inf
    epoch = 0
    log_epoch = 100
    max_epoch = 2000
    max_patience = 50

    x = model.g.ndata.pop("feat")
    for epoch in range(max_epoch):
        model.train()
        optimizer.zero_grad()
        loss = model.loss(x, target_ntype, train_nodes, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()
        if epoch % log_epoch == 0:
            logger.info(f"Epoch: {epoch}\t Loss: {loss:.4f}")

        # Validation
        model.eval()
        logits = model.forward(x, target_ntype)
        train_acc = accuracy(logits[train_nodes], train_labels)
        train_loss = model.cross_entropy_loss(logits[train_nodes], train_labels).cpu().item()
        val_acc = accuracy(logits[val_nodes], val_labels)
        val_loss = model.cross_entropy_loss(logits[val_nodes], val_labels).cpu().item()
        if epoch % log_epoch == 0:
            logger.info(f"Train: {train_acc:.3f}, {train_loss:.3f}, Val: {val_acc:.3f}, {val_loss:.3f}")
        if val_loss <= min_loss or val_acc >= max_score:
            if val_acc >= best_score:
                best_score = val_acc
                best_model = copy.deepcopy(model.state_dict())
            min_loss = np.min((min_loss, val_loss))
            max_score = np.max((max_score, val_acc))
            patience = 0
        else:
            patience += 1
            if patience == max_patience:
                model.load_state_dict(best_model)
                break

    # Test
    model.eval()
    logits = model.forward(x, target_ntype)
    test_acc = accuracy(logits[test_nodes], test_labels)
    logger.info(f"Test ACC = {test_acc}")

    torch.save(
        {
            "epoch": epoch,
            "model_type": HGN_TYPE,
            "optimizer": optimizer,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        hgn_path,
    )