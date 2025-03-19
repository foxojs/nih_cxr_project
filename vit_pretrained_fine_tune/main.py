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
from model_architectures import VisionTransformerPretrained
from evaluation import multi_label_evaluation
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import classification_report
from custom_datasets import nih_cxr_datamodule
import config 
from utils import save_config


ds_train = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", 'image-classification', split = "train[:1000]") 

# we hold back our test data to be used purely for testing, not in the context of our training loop 
ds_test = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", 'image-classification', split = "test[:500]") 


ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", 
                                          num_labels=15, 
                                          ignore_mismatched_sizes=True
                                          )


# continue using pytorch ligthing to fine tune model as per 

# https://towardsdatascience.com/how-to-fine-tune-a-pretrained-vision-transformer-on-satellite-data-d0ddd8359596/#:~:text=Under%20the%20hood%2C%20the%20trainer,is%20completed%20within%20few%20epochs.

def main(args): 
    L.seed_everything(42)


    # set up data 
    datamodule = nih_cxr_datamodule(batch_size=8)
    datamodule.prepare_data()
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    valid_dataloader = datamodule.valid_dataloader()
    test_dataloader = datamodule.test_dataloader()

    # setup model 
    model = VisionTransformerPretrained('google/vit-base-patch16-224', datamodule.num_classes, learning_rate= 1e-4)

    early_stopping = EarlyStopping(monitor = 'exact_accuracy', patience = 6, mode = 'max')

    logger = CSVLogger("tensorboard_logs", name = 'nih_cxr_pretrained_vit')

    log_dir = logger.log_dir

    #train 
    trainer = L.Trainer(devices = 1, max_epochs = config.NUM_EPOCHS, callbacks = [early_stopping], logger =logger)
    trainer.fit(model = model, train_dataloaders=train_dataloader, val_dataloaders = valid_dataloader)

    # evaluate on the test set 
    
    # we want to set our threshold based on micro average due to class imbalance - use micro average f1 score 

    multi_label_evaluation(model, test_dataloader = test_dataloader, 
                           test_dataset = ds_test, logger = logger)
    
    save_config(log_dir)
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
