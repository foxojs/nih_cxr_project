from sklearn.metrics import classification_report, roc_auc_score, f1_score, roc_curve, precision_recall_curve, auc
import torch 
import os 
import numpy as np 
from tqdm import tqdm 
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from utils import IdentityCalibrator

def calibrate_vision_transformer(valid_dataloader, best_model, logger, device): 
    """
    Fits a logistic regression model to our validation data using outputs from best model: 

    - Inference on the validation data 
    - Obtain logits for validation data 
    - pass logits as x in to logistic regression in a one v rest approach 
    - returns: multiple fitted logistic regression model for each label and also calibrated 
    
    """

    best_model.eval()
    log_dir = logger.log_dir
    os.makedirs(log_dir, exist_ok = True)

    all_true_labels = []
    all_pred_probs = []

    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc = "collecting logits"):
            x, y = batch 
            x = x.to(device)
            y = y.to(device)
            logits = best_model(x)
            all_pred_probs.append(torch.sigmoid(logits).cpu().numpy())
            all_true_labels.append(y.cpu().numpy())

    all_true_labels = np.vstack(all_true_labels)
    all_pred_probs = np.vstack(all_pred_probs)

    num_classes = all_true_labels.shape[1]
    calibrators = []
    for i in range(num_classes): 

        X = all_pred_probs[:, i].reshape(-1, 1)
        y = all_true_labels[:, i]

        if len(np.unique(y)) < 2:
            print(f"Label {i}: only one class in validation — using identity calibrator.")
            calibrator = IdentityCalibrator()
            calibrator.fit(X, y)
        else:
            calibrator = LogisticRegression(solver="lbfgs")
            calibrator.fit(X, y)
        
        calibrators.append(calibrator)

    return calibrators 


def multi_label_evaluation(device, 
                           model, 
                           test_dataloader, 
                           test_dataset, 
                           logger, 
                           calibrators: list[LogisticRegression] = None): 
    model.eval()
    log_dir = logger.log_dir
    os.makedirs(log_dir, exist_ok = True)

    all_true_labels = []
    all_pred_probs = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc = "Collecting logits"): 
            x, y = batch 
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_true_labels.append(y.cpu().numpy())
            all_pred_probs.append(probs) # store these so we can assess different thresholds quickly 

    all_true_labels = np.vstack(all_true_labels)
    all_pred_probs = np.vstack(all_pred_probs)

    pd.DataFrame(all_pred_probs, columns=test_dataset.features['labels'].feature.names).to_csv(os.path.join(log_dir, "all_pred_probs.csv"), 
                                                                                               index = False)

    label_list = test_dataset.features['labels'].feature.names

    # compute auc per label (remember labels are in the same order as our 1d one hot encoded array)

    # if calibrators provided, transform each columns 
    if calibrators is not None: 
        calibrated = np.zeros_like(all_pred_probs)

        for i, lr in enumerate(calibrators): 
            X = all_pred_probs[:, i].reshape(-1, 1)

            calibrated[:, i] = lr.predict_proba(X)[:, 1]
            print(f"This is calibrated {calibrated}")
        all_pred_probs_calibrated = calibrated 
        
    pd.DataFrame(all_pred_probs_calibrated, 
                 columns=test_dataset.features['labels'].feature.names).to_csv(os.path.join(log_dir, "all_pred_probs_calibrated.csv"), 
                                                                               index = False)


    auc_per_label = {}
    pr_auc_per_label = {}
    f1_per_label = {}
    best_thresholds = {}
    thresholds_to_test = np.linspace(0, 1, 10)


    for i, label in tqdm(enumerate(label_list), desc="Evaluating AUC, ROC, PR"):
        try:
            # ROC 
            fpr, tpr, _ = roc_curve(all_true_labels[:, i], all_pred_probs[:, i])
            auc_per_label[label] = roc_auc_score(all_true_labels[:, i], all_pred_probs[:, i])
            pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(os.path.join(log_dir, f"roc_curve_{label}.csv"), index=False)
            
            # Precision-Recall curve (imbalanced dataset)
            precision, recall, _ = precision_recall_curve(all_true_labels[:, i], all_pred_probs[:, i])
            pr_auc_per_label[label] = auc(recall, precision)
            pd.DataFrame({'precision': precision, 'recall': recall}).to_csv(os.path.join(log_dir, f"pr_curve_{label}.csv"), index=False)

            # find best threshold using f1 score 
            best_f1, best_t = 0, 0 
            for t in thresholds_to_test:
                preds = (all_pred_probs[:, i] >= t).astype(int)
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
    all_pred_labels = np.zeros_like(all_pred_probs)
    for i, label in enumerate(label_list):
        all_pred_labels[:, i] = (all_pred_probs[:, i] >= best_thresholds[label]).astype(int)


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


    # ───── Per‑label cardinality threshold calibration ─────
    true_freq = all_true_labels.mean(axis=0)
    card_thresholds = {}

    for i, label in enumerate(label_list):
        best_t, min_diff = 0.0, float("inf")
        for t in thresholds_to_test:
            pred_freq = (all_pred_probs[:, i] >= t).mean()
            diff = abs(pred_freq - true_freq[i])
            if diff < min_diff:
                best_t, min_diff = t, diff
        card_thresholds[label] = best_t

    # Save calibrated thresholds
    pd.DataFrame.from_dict(card_thresholds, orient='index', columns=['cardinality_threshold']) \
    .to_csv(os.path.join(log_dir, "cardinality_thresholds_per_label.csv"))

    # Generate predictions using these thresholds
    all_pred_labels_card = np.zeros_like(all_pred_probs, dtype=int)
    for i, label in enumerate(label_list):
        all_pred_labels_card[:, i] = (all_pred_probs[:, i] >= card_thresholds[label]).astype(int)

    # Save a new classification report
    report_card = classification_report(
        all_true_labels, all_pred_labels_card,
        target_names=label_list, zero_division=0, output_dict=True
    )
    pd.DataFrame(report_card).to_csv(os.path.join(log_dir, "test_multi_metrics_cardinality.csv"))







def multi_label_evaluation_from_checkpoint(device, model, test_dataloader, test_dataset, save_dir): 
    model.eval()
    log_dir = save_dir
    os.makedirs(log_dir, exist_ok = True)

    all_true_labels = []
    all_pred_probs = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc = "Collecting logits"): 
            x, y = batch 
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_true_labels.append(y.cpu().numpy())
            all_pred_probs.append(probs) # store these so we can assess different thresholds quickly 

    all_true_labels = np.vstack(all_true_labels)
    all_pred_probs = np.vstack(all_pred_probs)

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
            fpr, tpr, _ = roc_curve(all_true_labels[:, i], all_pred_probs[:, i])
            auc_per_label[label] = roc_auc_score(all_true_labels[:, i], all_pred_probs[:, i])
            pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv(os.path.join(log_dir, f"roc_curve_{label}.csv"), index=False)
            
            # Precision-Recall curve (imbalanced dataset)
            precision, recall, _ = precision_recall_curve(all_true_labels[:, i], all_pred_probs[:, i])
            pr_auc_per_label[label] = auc(recall, precision)
            pd.DataFrame({'precision': precision, 'recall': recall}).to_csv(os.path.join(log_dir, f"pr_curve_{label}.csv"), index=False)

            # find best threshold using f1 score 
            best_f1, best_t = 0, 0 
            for t in thresholds_to_test:
                preds = (all_pred_probs[:, i] >= t).astype(int)
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
    all_pred_labels = np.zeros_like(all_pred_probs)
    for i, label in enumerate(label_list):
        all_pred_labels[:, i] = (all_pred_probs[:, i] >= best_thresholds[label]).astype(int)


    # Save Classification Report
    final_report = classification_report(all_true_labels, all_pred_labels, target_names=label_list, zero_division=0, output_dict=True)
    pd.DataFrame(final_report).to_csv(os.path.join(log_dir, "test_multi_metrics_per_label_threshold.csv"))

    pd.DataFrame(all_pred_labels).to_csv(os.path.join(log_dir, "all_pred_labels.csv"))
    pd.DataFrame(all_true_labels).to_csv(os.path.join(log_dir, "all_true_labels.csv"))

    # and the probs 
    pd.DataFrame(all_pred_probs).to_csv(os.path.join(log_dir, "all_pred_probs.csv"))
    # Compute and Save Exact Match Accuracy
    exact_match_accuracy = accuracy_score(all_true_labels, all_pred_labels)
    with open(os.path.join(log_dir, "exact_match_accuracy.txt"), "w") as f:
        f.write(f"Exact Match Accuracy: {exact_match_accuracy:.4f}\n")
    print(f"Exact Match Accuracy: {exact_match_accuracy:.4f}")


    # ───── Per‑label cardinality threshold calibration ─────
    true_freq = all_true_labels.mean(axis=0)
    card_thresholds = {}

    for i, label in enumerate(label_list):
        best_t, min_diff = 0.0, float("inf")
        for t in thresholds_to_test:
            pred_freq = (all_pred_probs[:, i] >= t).mean()
            diff = abs(pred_freq - true_freq[i])
            if diff < min_diff:
                best_t, min_diff = t, diff
        card_thresholds[label] = best_t

    # Save calibrated thresholds
    pd.DataFrame.from_dict(card_thresholds, orient='index', columns=['cardinality_threshold']) \
    .to_csv(os.path.join(log_dir, "cardinality_thresholds_per_label.csv"))

    # Generate predictions using these thresholds
    all_pred_labels_card = np.zeros_like(all_pred_probs, dtype=int)
    for i, label in enumerate(label_list):
        all_pred_labels_card[:, i] = (all_pred_probs[:, i] >= card_thresholds[label]).astype(int)

    # Save a new classification report
    report_card = classification_report(
        all_true_labels, all_pred_labels_card,
        target_names=label_list, zero_division=0, output_dict=True
    )
    pd.DataFrame(report_card).to_csv(os.path.join(log_dir, "test_multi_metrics_cardinality.csv"))




    # now create relevant graphs


