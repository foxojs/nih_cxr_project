from datasets import load_dataset
from tqdm import tqdm
from datasets import Dataset
import torch 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from utils import extract_patches, MultiLabelDataset, ViT, TransformerBlock
import torch.optim as optim 
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm.notebook import trange, tqdm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
import torch
import numpy as np
import os

#If you haven't downloaded already, this takes aroun  40GB and around half an hour to download all the data (note that we split here)
# it will be cached on your system, so you access it in the future by running this code (rather than opening the files per se)
# once you've downloaded, hugging face automatically checks to see if you've already downloaded so subsequent loads are quick 

# we use just a small portion here to get our code working 
# hugging face has already split in to train and test so we can use train here and make a validation subset in our custom dataset class 
ds_train = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", 'image-classification', split = "train[:2500]") 

# we hold back our test data to be used purely for testing, not in the context of our training loop 
ds_test = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", 'image-classification', split = "test[:2000]") 


# now we can produce our train and validation dataloaders which will be used in training later
training_dataset_class = MultiLabelDataset(ds_train, image_size = (128, 128)) # make sure to have the channel dimension
training_dataset_class.train_validation_split()

# set dataset class mode to train to generate a training split 
training_dataset_class.mode = "train"
print(f"the size of training data is: {len(training_dataset_class)}")
train_dataloader = DataLoader(training_dataset_class, batch_size = 4, shuffle = True)

# set dataset class mode to val to generate a validation split 

training_dataset_class.mode = "val"
print(f" the size of validation data is: {len(training_dataset_class)}")
val_dataloader = DataLoader(training_dataset_class, batch_size = 4, shuffle = True)



test_dataset_class = MultiLabelDataset(ds_test, image_size = (128, 128))
test_dataset_class.mode = "test"
test_dataloader = DataLoader(test_dataset_class, batch_size = 4, shuffle = True)


# create model and view the output 

batch = next(iter(train_dataloader))
patch_size = 4

train_images = batch['image']
train_labels = batch['labels']

# set channels_in to the number of channels of the dataset images (in our case we have 3 channels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ViT(image_size = train_images.shape[2], 
            channels_in = train_images.shape[1], 
            patch_size = patch_size, 
            hidden_size = 128, 
            num_layers = 8, 
            num_heads = 8).to(device)

# pass an image through the network to check it works

try:
    out = model(train_images.to(device))
    print("The model works with your data")
    print("Output shape:", out.shape)  # Optional: Print the output shape
except Exception as e:
    print("Error:", e)



# set up the optimizer 


learning_rate = 0.01
num_epochs = 3

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                    T_max = num_epochs, 
                                                    eta_min = 0)

loss_fn = nn.BCEWithLogitsLoss() # note that we use this as we have multiple labels 

# define the training process ---------------------------

def train(model, optimizer, loader, device, loss_fn, loss_logger):

    # set network in train mode

    model.train()

    total_loss = 0
    num_batches = 0

    for i, batch in enumerate(tqdm(loader, leave = False, desc = "training")):
        # forward pass of image through network and get output 

        x = batch['image']
        y = batch['labels']

        fx = model(x.to(device))

        # calculate loss using loss function 

        loss = loss_fn(fx, y.float().to(device)) # this requires correct float 

        # zero gradients 

        optimizer.zero_grad()

        # back propagate

        loss.backward()

        # single optimisation step 

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        
    avg_loss = total_loss / num_batches if num_batches > 0 else 0  # Compute epoch loss
    loss_logger.append(avg_loss)  # Store epoch loss

    return model, optimizer, loss_logger


# define the testing process

def exact_match_accuracy(y_true, y_pred):
    """
    Computes the exact match accuracy.
    A sample is counted as correct only if all labels match exactly.
    """
    return (y_true == y_pred).all(dim=1).float().mean().item()

# This function should perform a single evaluation epoch, it WILL NOT be used to train our model
def evaluate(model, device, loader):
    
    # Initialise counter
    epoch_acc = 0
    
    # Set network in evaluation mode
    # Layers like Dropout will be disabled
    # Layers like Batchnorm will stop calculating running mean and standard deviation
    # and use current stored values (More on these layer types soon!)
    model.eval()
    
    with torch.no_grad():
    
        epoch_predicted_labels = []
        epoch_ground_truth_labels = []

        for i, batch in enumerate(tqdm(loader, leave=False, desc="Evaluating")):

                x = batch['image']
                y = batch['labels']
                # Forward pass of image through network
                fx = model(x.to(device))
                
                preds = (torch.sigmoid(fx) > 0.5).float()
                # Log the cumulative sum of the acc

                epoch_predicted_labels.append(preds.cpu().numpy())
                epoch_ground_truth_labels.append(y.cpu().numpy())

        # Concatenate all batches
        y_true_np = np.vstack(epoch_ground_truth_labels)
        y_pred_np = np.vstack(epoch_predicted_labels)

        exact_acc = accuracy_score(y_true_np, y_pred_np)
                                             
                    


            
    # Return the accuracy from the epoch     
    return exact_acc



# now the training process 

training_loss_logger = []
validation_acc_logger = []
training_acc_logger = []
best_val_accuracy = 0
best_model_path = "../trained_models/best_model.pth"  # Path to save the best model



# this implements training loop 

pbar = trange(0, num_epochs, leave= False, desc = "epoch")

for epoch in pbar: 
    valid_acc = 0
    train_acc = 0

    model, optimizer, training_loss_logger = train(model = model, 
                                                   optimizer = optimizer, 
                                                   loader = train_dataloader,
                                                   device = device, 
                                                   loss_fn = loss_fn, 
                                                   loss_logger = training_loss_logger
                                                   )
    
    # call evaluate function and pass dataloader for both validaiton and training 

    train_acc = evaluate(model = model, device = device, loader = train_dataloader)
    valid_acc = evaluate(model = model, device = device, loader = val_dataloader) # note we are using exact match accuracy 

    

    # log the train and validation accuracies 

    validation_acc_logger.append(valid_acc)
    training_acc_logger.append(train_acc)

    if valid_acc > best_val_accuracy:
        best_val_accuracy = valid_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with validation accuracy: {valid_acc:.4f}")

    # reduce the learning rate 

    lr_scheduler.step()

    pbar.set_postfix_str("Accuracy: Train %.2f%%, Val %.2f%%" % (train_acc * 100, valid_acc * 100))

print("Training complete")

dict = {"training_loss": training_loss_logger, 
        "validation_accuracy": validation_acc_logger, 
        "training_accuracy": training_acc_logger}

training_logs = pd.DataFrame(dict)
training_logs.to_csv(training_logs, "../results/training_logs.csv")




# Get class label mapping from Hugging Face dataset
label_list = ds_test.features['labels'].feature.names  # List of string labels
num_classes = len(label_list)

# Initialize lists to store all predictions and labels
all_true_labels = []
all_pred_labels = []


model = ViT(image_size = train_images.shape[2], 
            channels_in = train_images.shape[1], 
            patch_size = patch_size, 
            hidden_size = 128, 
            num_layers = 8, 
            num_heads = 8).to(device)

# Load the saved state dictionary
model.load_state_dict(torch.load("../trained_models/best_model.pth", map_location=device))

model.to(device)

model.eval()

# Loop over all batches in the dataloader
for batch in tqdm(test_dataloader, desc="Processing Batches"):
    
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
