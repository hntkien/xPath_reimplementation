from sklearn.metrics import precision_score, recall_score, f1_score 
from synconfig import DATASET_CONFIG, PATHS, logger
from utils import load_ground_truth_causes
import json

# ========= Evalluation Function ========== # 
def evaluate_explainer_predictions(
        prediced_dict: dict, 
        ground_truth_dict: dict, 
        target_nodes: list,
) -> dict: 
    """Evaluates the explainer predictions using precision, recall, and F1. 

    Args:
        prediced_dict (dict[str, list[int]]): Target node ID -> predicted cause 
            node IDs. 
        ground_truth_dict (dict[str, list[int]]): Target node ID -> true cause 
            node IDs. 
        target_nodes (list): List of target node IDs. 

    Returns:
        dict: 
            - precision (float): Precision score. 
            - recall (float): Recall score. 
            - f1 (float): F1 score.
    """
    # y_true_all = [] 
    # y_pred_all = [] 
    precisions = [] 
    recalls = [] 
    f1s = [] 

    for target_id in target_nodes:
        tid = str(target_id) 

        true_causes = set(ground_truth_dict.get(tid, [])) 
        predicted_causes = set(prediced_dict.get(tid, [])) 

        all_nodes = true_causes.union(predicted_causes) 

        # for node in all_nodes:
        #     y_true_all.append(1 if node in true_causes else 0) 
        #     y_pred_all.append(1 if node in predicted_causes else 0) 

        y_true = [1 if node in true_causes else 0 for node in all_nodes]
        y_pred = [1 if node in predicted_causes else 0 for node in all_nodes]

        # Calculate precision, recall, and F1 score for the current target node
        precisions.append(
            precision_score(y_true, y_pred, zero_division=0, average="macro"))
        recalls.append(
            recall_score(y_true, y_pred, zero_division=0, average="macro"))
        f1s.append(
            f1_score(y_true, y_pred, zero_division=0, average="macro"))
        

    # precision = precision_score(
    #     y_true_all, y_pred_all, zero_division=0, average="macro") 
    # recall = recall_score(
    #     y_true_all, y_pred_all, zero_division=0, average="macro")
    # f1 = f1_score(
    #     y_true_all, y_pred_all, zero_division=0, average="macro")

    precision = sum(precisions) / len(precisions) if precisions else 0.0 
    recall = sum(recalls) / len(recalls) if recalls else 0.0
    f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "precision": precision*100,
        "recall": recall*100,
        "f1": f1*100,
    }

def compute_iou_scores(
        predicted_dict:dict, 
        ground_truth_dict: dict, 
) -> dict: 
    """Computes the Intersection over Union (IoU) scores for the predicted 
    and ground truth cause node IDs.

    Args:
        predicted_dict (dict): Predicted cause node IDs. 
        ground_truth_dict (dict): Ground truth cause node IDs. 

    Returns:
        dict: IoU scores for each target node.
    """
    iou_scores = {}

    for target_id, true_causes in ground_truth_dict.items():
        pred_causes = predicted_dict.get(target_id, [])
        intersection = len(set(true_causes).intersection(set(pred_causes)))
        union = len(set(true_causes).union(set(pred_causes)))

        if union == 0:
            iou_scores[target_id] = 0.0
        else:
            iou_scores[target_id] = intersection / union

    return iou_scores

def average_io(iou_scores: dict) -> float:
    """Computes the average IoU score from the IoU scores dictionary.

    Args:
        iou_scores (dict): IoU scores for each target node.

    Returns:
        float: Average IoU score.
    """
    total_iou = sum(iou_scores.values())
    num_nodes = len(iou_scores)

    if num_nodes == 0:
        return 0.0

    return total_iou / num_nodes

# ========== Main Execution ========== #
if __name__ == "__main__":
    gt_dict, target_nodes = load_ground_truth_causes(
        dataset_name=DATASET_CONFIG["dataset_name"],
    )
    print(f"Number of target nodes: {len(target_nodes)}")

    # Load the predicted explanations
    with open(PATHS["cause_node_dict_path"], "r") as file:
        predicted_dict = json.load(file)

    # Evaluate the predictions
    evaluation_results = evaluate_explainer_predictions(
        prediced_dict=predicted_dict,
        ground_truth_dict=gt_dict,
        target_nodes=target_nodes,
    )
    # Print the evaluation results
    # print(f"Evaluation Results: {evaluation_results}")
    logger.info(f"Evaluation Results: {evaluation_results}")

    # Compute IoU scores
    iou_scores = compute_iou_scores(
        predicted_dict=predicted_dict,
        ground_truth_dict=gt_dict,
    )
    mean_iou = average_io(iou_scores)
    # print(f"Mean IoU: {mean_iou*100:.4f}")
    logger.info(f"Mean IoU: {mean_iou*100:.4f}")
