import torch
import torch.nn as nn
from torchvision import models
import config

def load_pretrained_vit(num_classes, device, fine_tune_layers = config.FINE_TUNE_LAYERS): 

    if config.USE_PRETRAINED == True: 
        print(f"loading pretrained viT model: {config.MODEL_NAME}")
        model = models.vit_b_16(weights = "IMAGENET1K_V1")

        for param in model.parameters(): 
            param.requires_grad = False # start by freezing all layers 

        
        # unfreeze the last few layers 

        for param in list(model.parameters())[-fine_tune_layers:]:
            param.requires_grad = True

        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)

