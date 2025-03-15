# we have a hugging face dataset, so we now define a custom dataset class which processes this, ensures our labels are one hot encoded etc. 

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


class MultiLabelDataset(Dataset):
    """
    Initialize with a Hugging Face dataset that's already formatted as torch tensors. 
    Will convert to tensors if plain hugging face dataset. 
    Will handle the multi label nature of our data through one hot encoding 
    
    Args:
        hf_dataset: A Hugging Face dataset with 'image' and 'labels' columns
    """
      
    def __init__(self, hf_dataset, image_size):

        self.x_train, self.x_val, self.y_train, self.y_val = None, None, None, None
        self.mode = "train"

        hf_dataset = hf_dataset.with_format("torch")
        print(hf_dataset.format)

        self.processed_images = []
        self.processed_labels = []

        for sample in tqdm(hf_dataset, desc= "processing image files"):
            image = sample['image']

            # we resize our image if specified 
            
            image = F.interpolate(image.unsqueeze(0), size = image_size, mode = "bilinear").squeeze(0)


            if image.shape[0] == 4:
                image = torch.index_select(image, 0, torch.tensor([0]))

            # normalize pixel values 
            image = image/255

            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1) # for convolutional networks we need 3 channels, remove this line and line below if wanting 1024, 1024 shape 

            # image = image.permute(1, 2, 0) # channel dimension needs to be the last one, not first 

            labels = sample['labels']
            one_hot = torch.zeros(15, dtype = torch.long)
            one_hot[labels] = 1
            self.processed_images.append(image)
            self.processed_labels.append(one_hot)
    
    def train_validation_split(self):
        """
        Takes our training data and produces a validation set from our training data
        Ensures that we don't use our hugging face defined test set during the training process 
        Means that we will assess trained models on a totally separate test set to avoid overfitting on test set
        Note that we use a subset of train as a validation set 
        """
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.processed_images, self.processed_labels, test_size = 0.2,
                                                                              random_state = 42)
        

    def __len__(self):
        """ Returns the length of our training or validation set depending on mode """
        if self.mode == "train":
            return len(self.x_train)
        elif self.mode == "val":
            return len(self.x_val)
        elif self.mode == "test":
            return len(self.processed_images)
    
        

    # note we are not doing lazy processing, so our data is processed when the dataset is instantiated. 
    # here we will return a train test split of our hugging face training data, unless we're using the test data in which case we return it all 
    def __getitem__(self, idx):
        """Gets items from either the training or validation set depending on mode"""
        if self.mode == "train":
            return {"image": self.x_train[idx], "labels": self.y_train[idx]}
        elif self.mode == "val":
            return {"image": self.x_val[idx], "labels": self.y_val[idx]}
        elif self.mode == "test":
            return {"image": self.processed_images[idx], "labels": self.processed_labels[idx]}
    
        
