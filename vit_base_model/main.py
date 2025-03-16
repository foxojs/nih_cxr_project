import os
import logging
import torch
import pandas as pd
import config
from tqdm import trange
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm 
import numpy as np 
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import matplotlib.pyplot as plt 
import seaborn as sns 

# Import custom modules
from dataset import MultiLabelDataset
from model import ViT
from train import train, evaluate

# Function to create an incrementing results folder
def create_results_folder():
    os.makedirs(config.RESULTS_PATH, exist_ok=True)
    new_run = max(
        (int(f.split('_')[-1]) for f in os.listdir(config.RESULTS_PATH) 
         if f.startswith("results_") and f.split('_')[-1].isdigit()), default=0) + 1
    new_folder = os.path.join(config.RESULTS_PATH, f"results_{new_run}")
    os.makedirs(new_folder, exist_ok=True)
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



# Function to prepare dataset and dataloaders
def prepare_dataloaders():
    ds_train = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", 'image-classification', split="train[:500]")
    dataset = MultiLabelDataset(ds_train, image_size=(128, 128))
    dataset.train_validation_split()

    dataset.mode = "train"
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    dataset.mode = "val"
    val_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    return train_loader, val_loader

# Main training function
def train_model(results_folder, device):
    # Set up paths
    log_file = os.path.join(results_folder, "training.log")
    logger = setup_logging(log_file)
    logger.info(f"Results are being saved to: {results_folder}")

    best_model_path = os.path.join(results_folder, "best_model.pth")

    # Load data
    train_loader, val_loader = prepare_dataloaders()
    logger.info(f"Training data size: {len(train_loader.dataset)}")
    logger.info(f"Validation data size: {len(val_loader.dataset)}")

    # Initialize model, optimizer, and loss function
    model = ViT().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=0)
    loss_fn = nn.BCEWithLogitsLoss()

    # Training process
    training_loss_logger = []
    validation_acc_logger = []
    training_acc_logger = []
    best_val_accuracy = 0

    pbar = trange(0, config.NUM_EPOCHS, leave=True, desc="Epoch")

    for epoch in pbar:
        # Train
        model, optimizer, training_loss_logger = train(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
            loss_fn=loss_fn,
            loss_logger=training_loss_logger
        )

        # Evaluate
        train_acc = evaluate(model=model, device=device, loader=train_loader)
        valid_acc = evaluate(model=model, device=device, loader=val_loader)

        # Log results
        training_acc_logger.append(train_acc)
        validation_acc_logger.append(valid_acc)
        pbar.set_postfix_str(f"Train Acc: {train_acc:.2%}, Val Acc: {valid_acc:.2%}")
        logger.info(f"Epoch {epoch+1}: Train Acc: {train_acc:.4f}, Val Acc: {valid_acc:.4f}")

        # Save best model
        if valid_acc > best_val_accuracy:
            best_val_accuracy = valid_acc
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with validation accuracy: {valid_acc:.4f}")

        # Adjust learning rate
        lr_scheduler.step()

    logger.info("Training complete")

    # Save logs
    training_logs = pd.DataFrame({
        "training_loss": training_loss_logger,
        "validation_accuracy": validation_acc_logger,
        "training_accuracy": training_acc_logger
    })
    training_logs.to_csv(os.path.join(results_folder, "training_logs.csv"), index=False)
    logger.info(f"Training logs saved to {results_folder}/training_logs.csv")


def plot_and_save_confusion_matrix(cm, class_name, save_path):
    """
    Plots and saves a single confusion matrix with labels.
    """
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix for {class_name}")

    # Save the figure
    file_name = f"{save_path}/confusion_matrix_{class_name}.png"
    plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.close()  # Close the figure to free memory


def compute_metrics(label_list, multi_cm, all_true_labels, all_pred_labels):
    # Initialize lists to store per-class metrics
    precision_list, recall_list, f1_list, accuracy_list = [], [], [], []

    # Compute per-class precision, recall, f1-score, accuracy
    cm_data = []
    for i, label in enumerate(label_list):
        tn, fp, fn, tp = multi_cm[i].ravel()  # Extract TN, FP, FN, TP

        # save the tn, fp, fn, tp for each class in a df 
        cm_data.append([label, tp, fp, tn, fn])

        # Compute Metrics
        precision = tp / (tp + fp + 1e-8)  # Avoid division by zero
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        # Append to lists
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        accuracy_list.append(accuracy)

    # **Compute Exact Match Accuracy**
    exact_match = accuracy_score(all_true_labels, all_pred_labels)  # Computes exact match accuracy

    # Create DataFrame
    cm_data = pd.DataFrame(cm_data, columns = ['Class', 'tp', 'fp', 'tn', 'fn'])

    df_metrics = pd.DataFrame({
        "Class": label_list,
        "Precision": precision_list,
        "Recall": recall_list,
        "F1-Score": f1_list,
        "Accuracy": accuracy_list
    })
    df_metrics.loc[len(df_metrics)] = ["Exact Match Accuracy", "", "", "", exact_match]


    return df_metrics, cm_data

def evaluate_model(results_folder, device):
    logger = setup_logging(os.path.join(results_folder, "evaluation.log"))
    logger.info("Starting evaluation...")

    # load test dataset
    ds_test = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", 'image-classification', split="test[:100]")
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
    all_pred_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, leave = True, desc = "Processing test batches"):
            batch_image = batch['image']
            batch_labels = batch['labels']

            fx = model(batch_image.to(device))
            pred = (torch.sigmoid(fx) > 0.5).float()
            all_true_labels.append(batch_labels.cpu().numpy())
            all_pred_labels.append(pred.cpu().numpy())

        # convert lists to full numpy arrays 
    all_true_labels = np.vstack(all_true_labels)
    all_pred_labels = np.vstack(all_pred_labels)

    multi_cm = multilabel_confusion_matrix(all_true_labels, all_pred_labels)
    plots_folder = os.path.join(results_folder, "plots/confusion_matrices")
    os.makedirs(plots_folder, exist_ok = True)

    for label, cm in zip(label_list, multi_cm):
        plot_and_save_confusion_matrix(cm, label, plots_folder)

    logger.info(f"confusion matrices saved in: {plots_folder}")

    # compute evaluation matrices 
    df_metrics, cm_data = compute_metrics(label_list, multi_cm, all_true_labels, all_pred_labels)

    # save metrics 
    metrics_folder = os.path.join(results_folder, "evaluation_metrics")
    os.makedirs(metrics_folder, exist_ok = True)
    metrics_file = os.path.join(metrics_folder, "df_metrics_overall.csv")
    cm_metrics_file = os.path.join(metrics_folder, "cm_metrics_file.csv")
    cm_data.to_csv(cm_metrics_file, index = False)
    df_metrics.to_csv(metrics_file, index = False)


    logger.info(f"Evaluation metrics saved to: {metrics_file}")
    logger.info("Evaluation process complete")


# Run the training process
if __name__ == "__main__":

    results_folder = create_results_folder()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_model(results_folder, device)

    evaluate_model(results_folder, device)
