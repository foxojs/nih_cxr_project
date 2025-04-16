import torch 
import tqdm 
import numpy as np 
from sklearn.metrics import f1_score
import config 
import os 
import pandas as pd
import logging
from datasets import load_dataset

def save_config(log_dir):
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
    print(f"the log directory is at: {log_dir}")
    os.makedirs(log_dir, exist_ok = True)

    config_file = os.path.join(log_dir, "run_configuration.csv")
    config_values.to_csv(config_file, index = False)


# to return just our raw probabilities if logistic regression can't be fit. 
class IdentityCalibrator:
    def fit(self, X, y):
        return self  # fit does nothing

    def predict_proba(self, X):
        # Ensure it returns shape (n_samples, 2) like sklearn
        return np.hstack([1 - X, X])