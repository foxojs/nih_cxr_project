from dataset import MultiLabelDataset
from sklearn.metrics import multilabel_confusion_matrix
from datasets import load_dataset
import torch 
import torch.nn as nn
from model import ViT
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import tqdm 
import os 
import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score
import config 
import seaborn as sns

# Device selection
if torch.backends.mps.is_available():  # Check for Apple MPS (Mac GPUs)
    device = torch.device("mps")
    print("Using Apple Metal (MPS) backend")
elif torch.cuda.is_available():  # Check for NVIDIA GPU (CUDA)
    device = torch.device("cuda")
    print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU (No GPU available)")

ds_test = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", 'image-classification', split = "test[:100]")

# Get class label mapping from Hugging Face dataset
label_list = ds_test.features['labels'].feature.names  # List of string labels
num_classes = len(label_list)

# Initialize lists to store all predictions and labels
all_true_labels = []
all_pred_labels = []

model = ViT()

# Load the saved state dictionary
model.load_state_dict(torch.load("../trained_models/best_model.pth", map_location=device))

model.to(device)

model.eval()

test_dataset_class = MultiLabelDataset(ds_test, image_size = config.IMAGE_SIZE)
test_dataset_class.mode = "test"
test_dataloader = DataLoader(test_dataset_class, batch_size = 4, shuffle = True)

# Loop over all batches in the dataloader
for batch in tqdm(test_dataloader, leave= True, desc="Processing Batches"):

    batch_image = batch['image']
    batch_labels = batch['labels']  # True labels

    with torch.no_grad():
        fx = model(batch_image.to(device))  # Forward pass
        pred = (torch.sigmoid(fx) > 0.5).float()  # Convert logits to binary labels

    # Store batch labels and predictions
    all_true_labels.append(batch_labels.cpu().numpy())  # Convert to NumPy and store
    all_pred_labels.append(pred.cpu().int().numpy())

# Convert lists to full NumPy arrays
all_true_labels = np.vstack(all_true_labels)  # Shape: (num_samples, num_classes)
all_pred_labels = np.vstack(all_pred_labels)  # Shape: (num_samples, num_classes)

# Compute multi-label confusion matrix
multi_cm = multilabel_confusion_matrix(all_true_labels, all_pred_labels)

# Function to plot confusion matrices with labels
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

# Define the directory where to save the files
save_directory = "../results/plots/confusion_matrices"

# Ensure the directory exists

os.makedirs(save_directory, exist_ok=True)

# Plot and save each confusion matrix with its class name
for label, cm in zip(label_list, multi_cm):
    plot_and_save_confusion_matrix(cm, label, save_directory)

print(f"Confusion matrices saved in: {save_directory}")


# Compute multi-label confusion matrix
multi_cm = multilabel_confusion_matrix(all_true_labels, all_pred_labels)

# Initialize lists to store per-class metrics
class_names = label_list  # Class labels from dataset
precision_list, recall_list, f1_list, accuracy_list = [], [], [], []

# Compute per-class precision, recall, f1-score, accuracy
for i, label in enumerate(class_names):
    tn, fp, fn, tp = multi_cm[i].ravel()  # Extract TN, FP, FN, TP

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
df_metrics = pd.DataFrame({
    "Class": class_names,
    "Precision": precision_list,
    "Recall": recall_list,
    "F1-Score": f1_list,
    "Accuracy": accuracy_list
})

# Add a row for Exact Match Accuracy (overall model accuracy)
df_metrics.loc[len(df_metrics)] = ["Exact Match Accuracy", "", "", "", exact_match]

df_metrics.to_csv("../results/df/df_metrics_overall.csv")
