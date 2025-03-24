from sklearn.metrics import classification_report, roc_auc_score, f1_score, roc_curve, precision_recall_curve, auc
import torch 
import os 
import numpy as np 
from tqdm import tqdm 
import pandas as pd 
from sklearn.metrics import accuracy_score

def multi_label_evaluation(device, model, test_dataloader, test_dataset, logger): 
    model.eval()
    log_dir = logger.log_dir
    os.makedirs(log_dir, exist_ok = True)

    all_true_labels = []
    all_pred_logits = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc = "Collecting logits"): 
            x, y = batch 
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_true_labels.append(y.cpu().numpy())
            all_pred_logits.append(probs) # store these so we can assess different thresholds quickly 

    all_true_labels = np.vstack(all_true_labels)
    all_pred_logits = np.vstack(all_pred_logits)

    label_list = test_dataset.features['labels'].feature.names

    # compute auc per label (remember labels are in the same order as our 1d one hot encoded array)

    auc_per_label = {}
    pr_auc_per_label = {}
    f1_per_label = {}
    best_thresholds = {}
    thresholds_to_test = np.linspace(0, 1, 10)


    for i, label in tqdm(enumerate(label_list), desc="Evaluating AUC, ROC, PR"):
        try:
            # ROC 
            fpr, tpr, _ = roc_curve(all_true_labels[:, i], all_pred_logits[:, i])
            auc_per_label[label] = roc_auc_score(all_true_labels[:, i], all_pred_logits[:, i])
            pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(os.path.join(log_dir, f"roc_curve_{label}.csv"), index=False)
            
            # Precision-Recall curve (imbalanced dataset)
            precision, recall, _ = precision_recall_curve(all_true_labels[:, i], all_pred_logits[:, i])
            pr_auc_per_label[label] = auc(recall, precision)
            pd.DataFrame({'precision': precision, 'recall': recall}).to_csv(os.path.join(log_dir, f"pr_curve_{label}.csv"), index=False)

            # find best threshold using f1 score 
            best_f1, best_t = 0, 0 
            for t in thresholds_to_test:
                preds = (all_pred_logits[:, i] >= t).astype(int)
                f1 = f1_score(all_true_labels[:, i], preds)
                if f1 > best_f1: 
                    best_f1 = f1
                    best_t = t
            
            best_thresholds[label] = best_t
            f1_per_label[label] = best_f1

        except ValueError:
            auc_per_label[label] = np.nan
            pr_auc_per_label[label] = np.nan
            best_thresholds[label] = np.nan
            f1_per_label[label] = np.nan # If AUC cannot be computed (e.g., only one class present)

    # Save AUC-ROC, PR AUC, Best Thresholds, and F1 Scores
    pd.DataFrame(list(auc_per_label.items()), columns=['label', 'auc_roc']).to_csv(os.path.join(log_dir, "auc_per_label.csv"), index=False)
    pd.DataFrame(list(pr_auc_per_label.items()), columns=['label', 'pr_auc']).to_csv(os.path.join(log_dir, "pr_auc_per_label.csv"), index=False)
    pd.DataFrame(list(best_thresholds.items()), columns=['label', 'best_threshold']).to_csv(os.path.join(log_dir, "best_thresholds_per_label.csv"), index=False)
    pd.DataFrame(list(f1_per_label.items()), columns=['label', 'best_f1_score']).to_csv(os.path.join(log_dir, "best_f1_scores_per_label.csv"), index=False)


    # Apply Best Thresholds to Generate Final Predictions
    all_pred_labels = np.zeros_like(all_pred_logits)
    for i, label in enumerate(label_list):
        all_pred_labels[:, i] = (all_pred_logits[:, i] >= best_thresholds[label]).astype(int)


    # Save Classification Report
    final_report = classification_report(all_true_labels, all_pred_labels, target_names=label_list, zero_division=0, output_dict=True)
    pd.DataFrame(final_report).to_csv(os.path.join(log_dir, "test_multi_metrics_per_label_threshold.csv"))

    pd.DataFrame(all_pred_labels).to_csv(os.path.join(log_dir, "all_pred_labels.csv"))
    pd.DataFrame(all_true_labels).to_csv(os.path.join(log_dir, "all_true_labels.csv"))
    # Compute and Save Exact Match Accuracy
    exact_match_accuracy = accuracy_score(all_true_labels, all_pred_labels)
    with open(os.path.join(log_dir, "exact_match_accuracy.txt"), "w") as f:
        f.write(f"Exact Match Accuracy: {exact_match_accuracy:.4f}\n")
    print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")


    # now create relevant graphs


def multi_label_evaluation_from_checkpoint(device, model, test_dataloader, test_dataset, save_dir): 
    model.eval()
    log_dir = save_dir
    os.makedirs(log_dir, exist_ok = True)

    all_true_labels = []
    all_pred_logits = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc = "Collecting logits"): 
            x, y = batch 
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_true_labels.append(y.cpu().numpy())
            all_pred_logits.append(probs) # store these so we can assess different thresholds quickly 

    all_true_labels = np.vstack(all_true_labels)
    all_pred_logits = np.vstack(all_pred_logits)

    label_list = test_dataset.features['labels'].feature.names

    # compute auc per label (remember labels are in the same order as our 1d one hot encoded array)

    auc_per_label = {}
    pr_auc_per_label = {}
    f1_per_label = {}
    best_thresholds = {}
    thresholds_to_test = np.linspace(0, 1, 10)


    for i, label in tqdm(enumerate(label_list), desc="Evaluating AUC, ROC, PR"):
        try:
            # ROC 
            fpr, tpr, _ = roc_curve(all_true_labels[:, i], all_pred_logits[:, i])
            auc_per_label[label] = roc_auc_score(all_true_labels[:, i], all_pred_logits[:, i])
            pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(os.path.join(log_dir, f"roc_curve_{label}.csv"), index=False)
            
            # Precision-Recall curve (imbalanced dataset)
            precision, recall, _ = precision_recall_curve(all_true_labels[:, i], all_pred_logits[:, i])
            pr_auc_per_label[label] = auc(recall, precision)
            pd.DataFrame({'precision': precision, 'recall': recall}).to_csv(os.path.join(log_dir, f"pr_curve_{label}.csv"), index=False)

            # find best threshold using f1 score 
            best_f1, best_t = 0, 0 
            for t in thresholds_to_test:
                preds = (all_pred_logits[:, i] >= t).astype(int)
                f1 = f1_score(all_true_labels[:, i], preds)
                if f1 > best_f1: 
                    best_f1 = f1
                    best_t = t
            
            best_thresholds[label] = best_t
            f1_per_label[label] = best_f1

        except ValueError:
            auc_per_label[label] = np.nan
            pr_auc_per_label[label] = np.nan
            best_thresholds[label] = np.nan
            f1_per_label[label] = np.nan # If AUC cannot be computed (e.g., only one class present)

    # Save AUC-ROC, PR AUC, Best Thresholds, and F1 Scores
    pd.DataFrame(list(auc_per_label.items()), columns=['label', 'auc_roc']).to_csv(os.path.join(log_dir, "auc_per_label.csv"), index=False)
    pd.DataFrame(list(pr_auc_per_label.items()), columns=['label', 'pr_auc']).to_csv(os.path.join(log_dir, "pr_auc_per_label.csv"), index=False)
    pd.DataFrame(list(best_thresholds.items()), columns=['label', 'best_threshold']).to_csv(os.path.join(log_dir, "best_thresholds_per_label.csv"), index=False)
    pd.DataFrame(list(f1_per_label.items()), columns=['label', 'best_f1_score']).to_csv(os.path.join(log_dir, "best_f1_scores_per_label.csv"), index=False)


    # Apply Best Thresholds to Generate Final Predictions
    all_pred_labels = np.zeros_like(all_pred_logits)
    for i, label in enumerate(label_list):
        all_pred_labels[:, i] = (all_pred_logits[:, i] >= best_thresholds[label]).astype(int)


    # Save Classification Report
    final_report = classification_report(all_true_labels, all_pred_labels, target_names=label_list, zero_division=0, output_dict=True)
    pd.DataFrame(final_report).to_csv(os.path.join(log_dir, "test_multi_metrics_per_label_threshold.csv"))

    pd.DataFrame(all_pred_labels).to_csv(os.path.join(log_dir, "all_pred_labels.csv"))
    pd.DataFrame(all_true_labels).to_csv(os.path.join(log_dir, "all_true_labels.csv"))
    # Compute and Save Exact Match Accuracy
    exact_match_accuracy = accuracy_score(all_true_labels, all_pred_labels)
    with open(os.path.join(log_dir, "exact_match_accuracy.txt"), "w") as f:
        f.write(f"Exact Match Accuracy: {exact_match_accuracy:.4f}\n")
    print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")


    # now create relevant graphs


