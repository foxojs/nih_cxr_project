from sklearn.metrics import classification_report
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

    # now we try and find our best threshold before classificaiton report at that threshold 
    thresholds_to_test = np.linspace(0, 1, 10)
    f1_micro_list = []

    for value in thresholds_to_test:
        predictions = (all_pred_logits >= value).astype(int)
        report = classification_report(all_true_labels, predictions, target_names = label_list, output_dict = True)
        micro_f1_average = report['micro avg']['f1-score']
        f1_micro_list.append({"Threshold": value, "Micro-F1": micro_f1_average})

    # save micro f1 scores 
    log_dir = logger.log_dir

    os.makedirs(log_dir, exist_ok = True)
    f1_micro_path = os.path.join(log_dir, "f1_micro_average.csv")
    f1_micro_average_df = pd.DataFrame(f1_micro_list)
    f1_micro_average_df.to_csv(f1_micro_path, index = False)
   
    # now select the threshold that gave the highest micro average 

    best_threshold = f1_micro_average_df.loc[f1_micro_average_df["Micro-F1"].idxmax(), "Threshold"]

    # calculate final labels based on best threshold 

    all_pred_labels = (all_pred_logits >= best_threshold).astype(int)
    
    report_path = os.path.join(log_dir, f"test_multi_metrics_{best_threshold:.4f}.csv")
    final_report = classification_report(all_true_labels, all_pred_labels, target_names=label_list, zero_division=0, output_dict = True)
    pd.DataFrame(final_report).to_csv(report_path)
