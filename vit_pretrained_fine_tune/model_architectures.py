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
import config
from torchmetrics.regression import MeanSquaredError

import lightning as L

class VisionTransformerPretrained(L.LightningModule): 
    '''wrapper for the pretrained vision transformers'''

    def __init__(self, model = "google/vit-base-patch16-224", 
                 num_classes = 15, 
                 learning_rate = config.LEARNING_RATE, 
                 test_dataset = None,
                 pos_weights = None):

        super().__init__()
        self.learning_rate = learning_rate 
        self.num_classes = num_classes
        self.backbone = ViTForImageClassification.from_pretrained(model, 
                                                             num_labels = num_classes, 
                                                             ignore_mismatched_sizes=True)
        
        if test_dataset is not None:
            self.label_names = test_dataset.features['labels'].feature.names
        else:
            self.label_names = [f"label_{i}" for i in range(num_classes)]

        if pos_weights is not None: 
            self.register_buffer("pos_weights", pos_weights, persistent=False)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
        else: 
            self.loss_fn = nn.BCEWithLogitsLoss()
        



        #metrics 

        self.mean_squared_error = MeanSquaredError()
        self.brier_per_label = MeanSquaredError(num_outputs=15) #brier score based on positive class i.e. minority class therefore better than just loss 
        self.multilabel_f1_weighted = torchmetrics.classification.MultilabelF1Score(num_labels = 15, threshold = 0.5, average = "weighted")
        self.multilabel_f1_micro = torchmetrics.classification.MultilabelF1Score(num_labels=15, threshold = 0.5, average = "micro")
        self.multilabel_f1_macro = torchmetrics.classification.MultilabelF1Score(num_labels = 15, threshold=0.5, average="macro")



    def forward(self, x): 
        return self.backbone(x).logits
    
    def step(self, batch, stage = "train"):
        '''Any step proccesses to return loss and predictions'''

        x, y = batch 

        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        y_hat = (torch.sigmoid(logits)>0.5).float() # we need to think a bit more about this perhaps to identify overfitting properly  

        loss = self.loss_fn(logits, y.float())

        self.mean_squared_error.update(probs, y)
        brier_score = self.mean_squared_error.compute()

        self.brier_per_label.update(probs, y)
        brier_per_label = self.brier_per_label.compute()

        self.multilabel_f1_weighted.update(y_hat, y)
        multilabel_f1_weighted = self.multilabel_f1_weighted.compute()

        self.multilabel_f1_micro.update(y_hat, y)
        multilabel_f1_micro = self.multilabel_f1_micro.compute()

        self.multilabel_f1_macro.update(y_hat, y)
        multilabel_f1_macro = self.multilabel_f1_macro.compute()

        return loss, multilabel_f1_weighted, multilabel_f1_micro, multilabel_f1_macro, y_hat, y, brier_score, brier_per_label, stage
    
    def training_step(self, batch, batch_idx): 
        loss, multilabel_f1_weighted, multilabel_f1_micro, multilabel_f1_macro, y_hat, y, brier_score, brier_per_label, stage = self.step(batch, stage = "train")

        for label_name, score in zip(self.label_names, brier_per_label):
            self.log(f"{stage}_brier_{label_name}", score, on_epoch=True)

        
        self.log("train_loss", loss)
        self.log("brier_score", brier_score, on_epoch = True, on_step = False)
        self.log("multilabel_f1_weighted", multilabel_f1_weighted, on_epoch=True, on_step=False)
        self.log("multilabel_f1_micro", multilabel_f1_micro, on_epoch=True, on_step=False)
        self.log("multilabel_f1_macro", multilabel_f1_macro, on_epoch=True, on_step=False)

        return loss 
    
    def validation_step(self, batch, batch_idx): 
        loss, multilabel_f1_weighted, multilabel_f1_micro, multilabel_f1_macro, y_hat, y, brier_score, brier_per_label, stage = self.step(batch, stage = "val")

        self.log("brier_score", brier_score)
        for label_name, score in zip(self.label_names, brier_per_label):
            self.log(f"{stage}_brier_{label_name}", score, on_epoch=True)

        self.log("val_multilabel_f1_weighted", multilabel_f1_weighted, on_epoch = True, on_step = False)
        self.log("val_loss", loss, on_epoch=True, on_step=False)
        self.log("val_multilabel_f1_micro", multilabel_f1_micro, on_epoch=True, on_step=False)
        self.log("val_multilabel_f1_macro", multilabel_f1_macro, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer 