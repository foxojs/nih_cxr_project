from sklearn.metrics import classification_report, roc_auc_score, f1_score, roc_curve, precision_recall_curve, auc
import torch 
import os 
import numpy as np 
from tqdm import tqdm 
import pandas as pd 
from sklearn.metrics import accuracy_score
from model_architectures import VisionTransformerPretrained
from custom_datasets import nih_cxr_datamodule
import config 
from evaluation import multi_label_evaluation_from_checkpoint
import lightning as L 
from datasets import load_dataset


def evaluate_from_checkpoint(version_to_evaluate, ckpt_name_to_evaluate = str(), batch_size = config.BATCH_SIZE):

    L.seed_everything(42, workers = True)

    if torch.backends.mps.is_available():  # Check for Apple MPS (Mac GPUs)
        device = torch.device("mps")
        print("Using Apple Metal (MPS) backend")
    elif torch.cuda.is_available():  # Check for NVIDIA GPU (CUDA)
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU (No GPU available)")


    save_dir = os.path.join(f"tensorboard_logs/nih_cxr_pretrained_vit/version_{version_to_evaluate}/checkpoint_evaluation")
    os.makedirs(save_dir, exist_ok = True)

    best_model_path = os.path.join(f"tensorboard_logs/nih_cxr_pretrained_vit/version_{version_to_evaluate}/{ckpt_name_to_evaluate}")


    best_model = VisionTransformerPretrained.load_from_checkpoint(best_model_path, strict=False)


    # set up data 
    datamodule = nih_cxr_datamodule(batch_size = batch_size)
    datamodule.prepare_data()
    datamodule.setup()

    test_dataloader = datamodule.test_dataloader()


    ds_test = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", 'image-classification', split = "test[:20]") # note this is just for labels 
    multi_label_evaluation_from_checkpoint(device, best_model, test_dataloader, test_dataset = ds_test, save_dir = save_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a Vision Transformer checkpoint on NIH CXR test data.")

    parser.add_argument(
        "--version", 
        type=str, 
        required=True, 
        help="Version number of the experiment (e.g., 3 for version_3)."
    )

    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        required=True, 
        help="Filename of the checkpoint to evaluate (e.g., 'epoch=4-step=9999.ckpt')."
    )

    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=config.BATCH_SIZE, 
        help="Batch size to use during evaluation (default from config)."
    )

    args = parser.parse_args()

    evaluate_from_checkpoint(
        version_to_evaluate=args.version,
        ckpt_name_to_evaluate=args.ckpt_path,
        batch_size=args.batch_size
    )



