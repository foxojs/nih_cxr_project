from datasets import load_dataset
from tqdm import tqdm
from datasets import Dataset
import torch 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
from torch.nn import functional as F
from torch import optim 
from transformers import ViTForImageClassification
import torchmetrics 
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
from torchvision.transforms import v2



ds_train = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", 'image-classification', split = "train[:500]") 

# we hold back our test data to be used purely for testing, not in the context of our training loop 
ds_test = load_dataset("alkzar90/NIH-Chest-X-ray-dataset", 'image-classification', split = "test[:500]") 



ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", 
                                          num_labels=15, 
                                          ignore_mismatched_sizes=True
                                          )


# continue using pytorch ligthing to fine tune model as per 

# https://towardsdatascience.com/how-to-fine-tune-a-pretrained-vision-transformer-on-satellite-data-d0ddd8359596/#:~:text=Under%20the%20hood%2C%20the%20trainer,is%20completed%20within%20few%20epochs.


class VisionTransformerPretrained(L.LightningModule): 
    '''wrapper for the pretrained vision transformers'''

    def __init__(self, model = "google/vit-base-patch16-224", num_classes = 1000):

        super().__init__()
        backbone = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", 
                                                             num_labels = 15, ignore_mismatched_sizes=True)
        
        self.backbone = backbone 

        self.loss_fn = nn.BCEWithLogitsLoss() # adjusted for our task of multilabel 

        #metrics 

        self.acc = torchmetrics.Accuracy("multilabel", num_classes=num_classes)

    def forward(self, x): 
        return self.backbone(x)
    
    def step(self, batch):
        '''Any step proccesses to return loss and predictions'''

        x, y = batch 

        prediction = self.backbone(x)
        y_hat = (torch.sigmoid(prediction.logits>0.5)).float() # we need to 

        loss = self.loss_fn(prediction.logits, y.float())
        acc = self.acc(y_hat, y)

        return loss, acc, y_hat, y
    
    def training_step(self, batch, batch_idx): 
        loss, acc, y_hat, y = self.step(batch)

        self.log("train_loss", loss)

        return loss 
    
    def validation_step(self, batch, batch_idx): 
        loss, acc, y_hat, y = self.step(batch)

        self.log("valid_acc", acc, on_epoch = True, on_step = False)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = 1e-4)
        return optimizer 
    

class HuggingFaceCXR(Dataset):
    """
    Custom PyTorch Dataset wrapper for Hugging Face datasets.
    Converts dataset samples into a PyTorch-compatible format.
    """
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]  # Adjust this key based on your dataset structure
        label = torch.tensor(item["labels"], dtype=torch.float32)  # Multi-label case

        if self.transform:
            image = self.transform(image)

        return image, label
    

class nih_cxr_datamodule(L.LightningDataModule):
    '''Lightning data module for the cxr dataset'''

    def __init__(self, batch_size, data_root="alkzar90/NIH-Chest-X-ray-dataset"): 
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size 

    def setup(self, stage = None):
        '''set up the dataset, train/valid/test all at once'''

        transforms = v2.Compose([v2.ToImage(),
                                 v2.Resize(size=(224,224), interpolation=2),
                                 v2.ToDtype(torch.float32, scale=True),
                                 v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                ])
        
        ds_train = load_dataset(self.data_root, 'image_classification', split = "train[:500]")

        train_valid_split = ds_train.train_test_split(test_size = 0.2, stratify_by_column='labels')

        ds_train = train_valid_split['train']
        ds_valid = train_valid_split['test']

 
        ds_test = load_dataset(self.data_root, 'image-classification', split = "test[:500]") 

        self.train_data = HuggingFaceCXR(ds_train, transform = transforms)
        self.valid_data = HuggingFaceCXR(ds_valid, transform = transforms)
        self.test_data = HuggingFaceCXR(ds_test, transform = transforms)


    def train_dataloader(self): 
        return DataLoader(self.train_data, batch_size = self.batch_size, shuffle = True)
    
    def valid_dataloader(self): 
        return DataLoader(dataset = self.valid_data, batch_size = self.batch_size, shuffle = False)
    
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size = self.batch_size, shuffle = False)
        


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

    early_stopping = EarlyStopping(monitor = 'valid_acc', patience = 6, mode = 'max')

    logger = CSVLogger("tensorboard_logs", name = 'nih_cxr_pretrained_vit')

    #train 
    trainer = L.Trainer(devices = 1, callbacks = [early_stopping], logger =logger)
    trainer.fit(model = model, train_dataloaders=train_dataloader, val_dataloaders = valid_dataloader)

    #test 
    trainer.test(model = model, dataloaders = test_dataloader, verbose = True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)