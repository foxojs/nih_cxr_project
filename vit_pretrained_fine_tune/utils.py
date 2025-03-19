import torch 
import tqdm 
import numpy as np 
from sklearn.metrics import f1_score
import config 
from dataset import MultiLabelDataset
import os 
import pandas as pd
import logging
from datasets import load_dataset

def save_config(results_folder):
    config_values = pd.DataFrame({
        "IMAGE_SIZE":config.IMAGE_SIZE,
        "PATCH_SIZE": config.PATCH_SIZE,
        "HIDDEN_SIZE": config.HIDDEN_SIZE,
        "NUM_LAYERS": config.NUM_LAYERS,
        "NUM_HEADS": config.NUM_HEADS,
        "NUM_CLASSES": config.NUM_CLASSES,
        "BATCH_SIZE": config.BATCH_SIZE,
        "LEARNING_RATE": config.LEARNING_RATE,
        "NUM_EPOCHS": config.NUM_EPOCHS
    })

    config_file = os.path.join(results_folder, "run_configuration.csv")
    config_values.to_csv(config_file, index = False)



