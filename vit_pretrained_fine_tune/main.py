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
from lightning.pytorch.callbacks import ModelCheckpoint


# continue using pytorch ligthing to fine tune model as per 

# https://towardsdatascience.com/how-to-fine-tune-a-pretrained-vision-transformer-on-satellite-data-d0ddd8359596/#:~:text=Under%20the%20hood%2C%20the%20trainer,is%20completed%20within%20few%20epochs.

def main(args): 
    L.seed_everything(42)

    if torch.backends.mps.is_available():  # Check for Apple MPS (Mac GPUs)
        device = torch.device("mps")
        print("Using Apple Metal (MPS) backend")
    elif torch.cuda.is_available():  # Check for NVIDIA GPU (CUDA)
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (No GPU available)")

    # set up data 
    datamodule = nih_cxr_datamodule(batch_size=8)
    datamodule.prepare_data()
    datamodule.setup()

    train_dataloader = datamodule.train_dataloader()
    valid_dataloader = datamodule.valid_dataloader()
    test_dataloader = datamodule.test_dataloader()

    # setup model 
    model = VisionTransformerPretrained('google/vit-base-patch16-224', datamodule.num_classes, learning_rate= 1e-4)

    logger = CSVLogger("tensorboard_logs", name = 'nih_cxr_pretrained_vit')

    log_dir = logger.log_dir

    checkpoint_callback = ModelCheckpoint(dirpath=log_dir, monitor = "val_multi_label_f1", 
                                          save_on_train_epoch_end=True, save_top_k = 2, mode="max",
                                          filename="best_model_{epoch}_{val_multi_label_f1}")

    #train 
    trainer = L.Trainer(devices = 1, max_epochs = config.NUM_EPOCHS, callbacks = [checkpoint_callback], logger =logger)
    trainer.fit(model = model, train_dataloaders=train_dataloader, val_dataloaders = valid_dataloader)

    # load the saved model before evaluating 

    
    # we want to set our threshold based on micro average due to class imbalance - use micro average f1 score 

    best_model_path = checkpoint_callback.best_model_path
    
    best_model = VisionTransformerPretrained.load_from_checkpoint(best_model_path)

    ds_test = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", 'image-classification', split = "test[:20]") # note this is just for labels 

    multi_label_evaluation(device, model = best_model, test_dataloader = test_dataloader, 
                           test_dataset = ds_test, logger = logger)
    
    save_config(log_dir)
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
