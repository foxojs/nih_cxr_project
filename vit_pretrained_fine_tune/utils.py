import torch 
import tqdm 
import numpy as np 
from sklearn.metrics import f1_score
import config 
from dataset import MultiLabelDataset
import os 
import pandas as pd
import logging
from datasets import load_dataset

def save_config(results_folder):
    config_values = pd.DataFrame({
        "IMAGE_SIZE":config.IMAGE_SIZE,
        "PATCH_SIZE": config.PATCH_SIZE,
        "HIDDEN_SIZE": config.HIDDEN_SIZE,
        "NUM_LAYERS": config.NUM_LAYERS,
        "NUM_HEADS": config.NUM_HEADS,
        "NUM_CLASSES": config.NUM_CLASSES,
        "BATCH_SIZE": config.BATCH_SIZE,
        "LEARNING_RATE": config.LEARNING_RATE,
        "NUM_EPOCHS": config.NUM_EPOCHS
    })

    config_file = os.path.join(results_folder, "run_configuration.csv")
    config_values.to_csv(config_file, index = False)


def create_results_folder():
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    new_run = max(
        (int(f.split('_')[-1]) for f in os.listdir(config.RESULTS_PATH) 
         if f.startswith("results_") and f.split('_')[-1].isdigit()), default=0) + 1
    new_folder = os.path.join(config.RESULTS_PATH, f"results_{new_run}")
    os.makedirs(new_folder, exist_ok=True)

    # save our configuration
    save_config(results_folder=new_folder)
    return new_folder

# Function to configure logging inside the results folder
def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()
    return logger



def evaluate_multi_label(results_folder, device):
    logger = setup_logging(os.path.join(results_folder, "evaluation.log"))
    logger.info("Starting evaluation...")
    logger.info(f"device is {device}")

    # load test dataset
    ds_test = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", 'image-classification', split=config.DS_TEST_SIZE)
    label_list = ds_test.features['labels'].feature.names

    model_path = os.path.join(results_folder, "best_model.pth")
    model = ViT()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # prepare test dataloader 
    test_dataset_class = MultiLabelDataset(ds_test, image_size=config.IMAGE_SIZE)
    test_dataset_class.mode = "test"
    test_dataloader = DataLoader(test_dataset_class, batch_size=config.BATCH_SIZE, shuffle = False)

    all_true_labels = []
    all_pred_probs = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, leave = True, desc = "Processing test batches"):
            batch_image = batch['image']
            batch_labels = batch['labels']

            fx = model(batch_image.to(device))
            probs = torch.sigmoid(fx).cpu().numpy()
            all_true_labels.append(batch_labels.cpu().numpy())
            all_pred_probs.append(probs)

        # convert lists to full numpy arrays 
    all_true_labels = np.vstack(all_true_labels)
    all_pred_probs = np.vstack(all_pred_probs)

    # now compute the roc/auc for each class 

    roc_results = {}
    optimal_thresholds = {}

    for i, label in enumerate(label_list):
        fpr, tpr, thresholds = roc_curve(all_true_labels[:, i], all_pred_probs[:, i])
        auc_score = auc(fpr, tpr)

        # find the threshold that maximises the f1 score 

        f1_scores = []
        for threshold in thresholds: 
            pred_labels = (all_pred_probs[:, i] >= threshold).astype(int)
            f1 = f1_score(all_true_labels[:, i], pred_labels)
            f1_scores.append(f1)

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_thresholds[label] = float(optimal_threshold)

        roc_results[label] = {
            "fpr":fpr.tolist(), 
            "tpr": tpr.tolist(), 
            "thresholds": thresholds.tolist(), 
            "auc": auc_score, 
            "optimal_threshold": float(optimal_threshold)
        }

    roc_folder = os.path.join(results_folder, "roc_data")
    os.makedirs(roc_folder, exist_ok = True)
    roc_file = os.path.join(roc_folder, "roc_results.json")

    with open(roc_file, "w") as f:
        json.dump(roc_results, f, indent = 4)

    logger.info(f"ROC data saved to: {roc_file}")

    # now we use the optimal threshold to compute the multi label confusion matrix and associated metrics for each class 

    all_pred_labels = np.zeros_like(all_pred_probs) #gives same array structure 
    for i, label in enumerate(label_list): 
        all_pred_labels[:, i] = (all_pred_probs[:, i] >= optimal_thresholds[label])

    # using this optimal threshold (for f1 score) compute the matrix 

    multi_cm = multilabel_confusion_matrix(all_true_labels, all_pred_labels)

    plots_folder = os.path.join(results_folder, "plots/confusion_matrices")
    os.makedirs(plots_folder, exist_ok = True)

    for label, cm in zip(label_list, multi_cm):
        plot_and_save_confusion_matrix(cm, label, plots_folder)

    logger.info(f"confusion matrices saved in: {plots_folder}")

    # compute evaluation matrices 
    df_metrics, cm_data = compute_metrics(label_list, multi_cm, all_true_labels, all_pred_labels, optimal_thresholds)

    # save metrics 
    metrics_folder = os.path.join(results_folder, "evaluation_metrics")
    os.makedirs(metrics_folder, exist_ok = True)
    metrics_file = os.path.join(metrics_folder, "df_metrics_overall.csv")
    cm_metrics_file = os.path.join(metrics_folder, "cm_metrics_file.csv")
    cm_data.to_csv(cm_metrics_file, index = False)
    df_metrics.to_csv(metrics_file, index = False)


    logger.info(f"Evaluation metrics saved to: {metrics_file}")
    logger.info("Evaluation process complete")
