from sklearn.metrics import classification_report, roc_auc_score, f1_score
import torch 
import os 
import numpy as np 
from tqdm import tqdm 
import pandas as pd 

def multi_label_evaluation(model, test_dataloader, test_dataset, logger): 
    model.eval()

    all_true_labels = []
    all_pred_logits = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc = "Collecting logits"): 
            x, y = batch 
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_true_labels.append(y.cpu().numpy())
            all_pred_logits.append(probs) # store these so we can assess different thresholds quickly 

    all_true_labels = np.vstack(all_true_labels)
    all_pred_logits = np.vstack(all_pred_logits)

    label_list = test_dataset.features['labels'].feature.names

    # compute auc per label (remember labels are in the same order as our 1d one hot encoded array)

    auc_per_label = {}
    for i, label in enumerate(label_list):
        try:
            auc_per_label[label] = roc_auc_score(all_true_labels[:, i], all_pred_logits[:, i])
        except ValueError:
            auc_per_label[label] = np.nan
    log_dir = logger.log_dir
    os.makedirs(log_dir, exist_ok = True)

    auc_path = os.path.join(log_dir, "auc_per_label.csv")
    auc_df = pd.DataFrame(list(auc_per_label.items()), columns=['label', 'auc_roc'])
    auc_df.to_csv(auc_path, index = False)



    # now we try and find our best threshold for each label 
    best_thresholds = {}
    thresholds_to_test = np.linspace(0, 1, 10)

    for i, label in enumerate(label_list):
        best_f1 = 0
        best_t = 0

        for t in thresholds_to_test:
            preds = (all_pred_logits[:, i] >= t).astype(int) # apply threshold t to class i 
            f1 = f1_score(all_true_labels[:, i], preds)
            
            if f1 > best_f1:
                best_f1 = f1
                best_t = t # store the best threshold for this class 
        
        best_thresholds[label] = best_t

    # save micro f1 scores 
    log_dir = logger.log_dir

    os.makedirs(log_dir, exist_ok = True)
    threshold_path = os.path.join(log_dir, "best_thresholds_per_label.csv")
    pd.DataFrame(list(best_thresholds.items()), columns = ['label', 'best_threshold']).to_csv(threshold_path, index = False)
   
    # now select the threshold that gave the highest micro average  

    all_pred_labels = np.zeros_like(all_pred_logits)
    for i, label in enumerate(label_list):
        all_pred_labels[:, i] = (all_pred_logits[:, i] >= best_thresholds[label]).astype(int)

    report_path = os.path.join(log_dir, "test_multi_metrics_per_label_threshold.csv")
    final_report = classification_report(all_true_labels, all_pred_labels, target_names=label_list, zero_division=0, output_dict=True)
    pd.DataFrame(final_report).to_csv(report_path)




