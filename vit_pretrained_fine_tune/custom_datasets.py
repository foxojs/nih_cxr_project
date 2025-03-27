from datasets import load_dataset
from tqdm import tqdm
from datasets import Dataset
import torch 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import tqdm
from tqdm import trange, tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
import os 
from PIL import Image
from transformers import ViTForImageClassification
from torch.nn import functional as F
from torch import optim 
import torchmetrics 
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2
import lightning as L
from torch.utils.data import DataLoader, Subset
from datasets import load_dataset
from tqdm import tqdm
from datasets import Dataset
import torch 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import tqdm
from tqdm import trange, tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
import os 
from PIL import Image
from transformers import ViTForImageClassification
from torch.nn import functional as F
from torch import optim 
import torchmetrics 
import lightning as L
from model_architectures import VisionTransformerPretrained
from evaluation import multi_label_evaluation
import config 



import lightning as L

class HuggingFaceCXR(Dataset):
    """
    Custom PyTorch Dataset wrapper for Hugging Face datasets.
    Converts dataset samples into a PyTorch-compatible format.
    """
    def __init__(self, hf_dataset, image_size=config.IMAGE_SIZE, num_classes=config.NUM_CLASSES, transform=None):
        self.dataset = hf_dataset.with_format("torch")  # Ensure dataset is in torch format
        self.image_size = image_size
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Load image
        image = item["image"]  # Ensure this matches your dataset keys

        # Resize image
        image = F.interpolate(image.unsqueeze(0), size = self.image_size, mode = "bilinear").squeeze(0)

        # Handle 4-channel (RGBA) images: Keep only RGB
        if image.shape[0] == 4:
            image = image[:3, :, :]

        # Handle Grayscale (1-channel) images: Convert to 3-channel
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        # Normalize pixel values to [0,1]
        image = image / 255.0

        # Convert labels to one-hot encoding
        labels = item["labels"]
        one_hot = torch.zeros(self.num_classes, dtype=torch.float32)
        one_hot[labels] = 1  # Set corresponding indices to 1

        # Apply optional transformations
        if self.transform:
            image = self.transform(image)

        return image, one_hot
    


class nih_cxr_datamodule(L.LightningDataModule):
    '''Lightning data module for the cxr dataset'''

    def __init__(self, batch_size=config.BATCH_SIZE, data_root="alkzar90/NIH-Chest-X-ray-dataset"): 
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size 
        self.num_classes = config.NUM_CLASSES

    @staticmethod
    def compute_pos_weights(dataset, num_classes): 
        counts = torch.zeros(num_classes)

        for i in range(len(dataset)): 
            labels = dataset[i]['labels']
            for label in labels: 
                counts[label] += 1
        
        total_samples = len(dataset)
        counts = torch.clamp(counts, min=1)
        pos_weights = (total_samples - counts) / counts
        return pos_weights 

    def setup(self, stage = None, image_augmentations = config.IMAGE_AUGMENTATION):
        '''set up the dataset, train/valid/test all at once'''

        if image_augmentations: 
            transforms = v2.Compose([v2.ToImage(),
                                 v2.Resize(size=config.IMAGE_SIZE, interpolation=2),
                                 v2.Grayscale(num_output_channels=3), # need to ensure 3 channel grayscale for vit 
                                 v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                 v2.RandAugment()
                                ])
        else: 
            transforms = v2.Compose([v2.ToImage(),
                                 v2.Resize(size=config.IMAGE_SIZE, interpolation=2),
                                 v2.Grayscale(num_output_channels=3), # need to ensure 3 channel grayscale for vit 
                                 v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])
        
        ds_train = load_dataset(self.data_root, 'image-classification', split = config.DS_TRAIN_SIZE)

        train_valid_split = ds_train.train_test_split(test_size = 0.2)

        ds_train = train_valid_split['train']
        ds_valid = train_valid_split['test']

 
        ds_test = load_dataset(self.data_root, 'image-classification', split = config.DS_TEST_SIZE) 

        self.train_data = HuggingFaceCXR(ds_train, image_size = config.IMAGE_SIZE, transform = transforms)
        self.valid_data = HuggingFaceCXR(ds_valid, image_size = config.IMAGE_SIZE, transform = transforms)
        self.test_data = HuggingFaceCXR(ds_test, image_size = config.IMAGE_SIZE, transform = transforms)

        # calculate pos_weight before wrapping in huggingface CXR
        self.pos_weights = self.compute_pos_weights(ds_train, self.num_classes)


    def train_dataloader(self): 
        return DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True, 
                          num_workers=config.NUM_WORKERS, persistent_workers=True)
    
    def val_dataloader(self): 
        return DataLoader(dataset = self.valid_data, batch_size = self.batch_size, shuffle = False, 
                          num_workers=config.NUM_WORKERS, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = self.batch_size, 
                          shuffle = False, num_workers=config.NUM_WORKERS, 
                          persistent_workers=True)
    
        
