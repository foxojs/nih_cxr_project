from datasets import load_dataset
import torchmetrics.classification
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
from torcheval.metrics import MultilabelAccuracy

import lightning as L

class VisionTransformerPretrained(L.LightningModule): 
    '''wrapper for the pretrained vision transformers'''

    def __init__(self, model = "google/vit-base-patch16-224", num_classes = 15, learning_rate = 1e-4):

        super().__init__()
        self.learning_rate = learning_rate 
        self.num_classes = num_classes
        backbone = ViTForImageClassification.from_pretrained(model, 
                                                             num_labels = num_classes, 
                                                             ignore_mismatched_sizes=True)
        
        self.backbone = backbone 

        self.loss_fn = nn.BCEWithLogitsLoss() # adjusted for our task of multilabel 

        #metrics 

        self.multilabel_f1 = torchmetrics.classification.MultilabelF1Score(num_labels = 15, threshold = 0.5, average = "weighted")



    def forward(self, x): 
        return self.backbone(x).logits
    
    def step(self, batch, stage = "train"):
        '''Any step proccesses to return loss and predictions'''

        x, y = batch 

        logits = self.forward(x)
        y_hat = (torch.sigmoid(logits)>0.5).float() # we need to think a bit more about this perhaps to identify overfitting properly  

        loss = self.loss_fn(logits, y.float())

        self.multilabel_f1.update(y_hat, y)
        multi_label_f1 = self.multilabel_f1.compute()

        return loss, multi_label_f1, y_hat, y
    
    def training_step(self, batch, batch_idx): 
        loss, val_multi_label_f1, y_hat, y = self.step(batch)

        self.log("train_loss", loss)

        return loss 
    
    def validation_step(self, batch, batch_idx): 
        loss, val_multi_label_f1, y_hat, y = self.step(batch)

        self.log("val_multi_label_f1", val_multi_label_f1, on_epoch = True, on_step = False)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer 
