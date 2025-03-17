import torch 
import tqdm 
import numpy as np 
from sklearn.metrics import f1_score
import config 
from dataset import MultiLabelDataset
import os 

def extract_patches(image_tensor, patch_size = 4):
    """Converts images to patches for input to a vision transformer"""

    
    bs, c, h, w = image_tensor.size()

    # define teh unfold layer with appropriate parameters 

    unfold = torch.nn.Unfold(kernel_size = patch_size, stride = patch_size)

    unfolded = unfold(image_tensor)

    # reshape the unfolded tensor to match the desired output shape 
    # output shaep BS x L x C x 8 x8 where L is the number of patches in each dimension 
    # fo reach dimension, number of patches = (original dimension size) //patch_size 

    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)

    return unfolded



def evaluate(model, device, loader):
    """evaluates a model and returns exact accuracy for a single epoch 
    args: 
    - model = a trained model 
    - device = where the data and model are stored 
    - loader = train/test/validation loader 
    """

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

        # we need to iterate over the loader to get the overall batch accuracy 

        for i, batch in enumerate(tqdm(loader, leave=True, desc="Evaluating")):

                x = batch['image']
                y = batch['labels']
                # Forward pass of image through network
                fx = model(x.to(device))

                # we are using a threshold of 0.5, probably something to tune 
                preds = (torch.sigmoid(fx) > 0.5).float()
                
                # Log the cumulative sum of the acc

                epoch_predicted_labels.append(preds.cpu().numpy())
                epoch_ground_truth_labels.append(y.cpu().numpy())

        # Concatenate all batches
        y_true_np = np.vstack(epoch_ground_truth_labels)
        y_pred_np = np.vstack(epoch_predicted_labels)

        macro_f1 = f1_score(y_true_np, y_pred_np, average = "macro")
    # Return the accuracy from the epoch
    return macro_f1

