import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import config
from sklearn.metrics import accuracy_score

def train(model, optimizer, loader, device, loss_fn, loss_logger):

    # set network in train mode

    model.train()

    total_loss = 0
    num_batches = 0

    for i, batch in enumerate(tqdm(loader, leave = True, desc = "training")):
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

from sklearn.metrics import accuracy_score
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

        for i, batch in enumerate(tqdm(loader, leave=True, desc="Evaluating")):

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
