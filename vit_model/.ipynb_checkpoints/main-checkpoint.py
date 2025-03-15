import torch
from dataset import MultiLabelDataset
from model import ViT
from train import train, evaluate
import config
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm, trange
from datasets import Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# now we can produce our train and validation dataloaders which will be used in training later
training_dataset_class = MultiLabelDataset(ds_train, image_size = (128, 128)) # make sure to have the channel dimension
training_dataset_class.train_validation_split()

# set dataset class mode to train to generate a training split 
training_dataset_class.mode = "train"
print(f"the size of training data is: {len(training_dataset_class)}")
train_dataloader = DataLoader(training_dataset_class, batch_size = 4, shuffle = True)

# set dataset class mode to val to generate a validation split 

training_dataset_class.mode = "val"
print(f" the size of validation data is: {len(training_dataset_class)}")
val_dataloader = DataLoader(training_dataset_class, batch_size = 4, shuffle = True)
model = ViT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
loss_fn = nn.BCEWithLogitsLoss()

# now the training process

training_loss_logger = []
validation_acc_logger = []
training_acc_logger = []
best_val_accuracy = 0
best_model_path = "../trained_models/best_model.pth"  # Path to save the best model



# this implements training loop

pbar = trange(0, num_epochs, leave= True, desc = "epoch")

for epoch in pbar:
    valid_acc = 0
    train_acc = 0

    model, optimizer, training_loss_logger = train(model = model,
                                                   optimizer = optimizer,
                                                   loader = train_dataloader,
                                                   device = device,
                                                   loss_fn = loss_fn,
                                                   loss_logger = training_loss_logger
                                                   )

    # call evaluate function and pass dataloader for both validaiton and training

    train_acc = evaluate(model = model, device = device, loader = train_dataloader)
    valid_acc = evaluate(model = model, device = device, loader = val_dataloader) # note we are using exact match accuracy



    # log the train and validation accuracies

    validation_acc_logger.append(valid_acc)
    training_acc_logger.append(train_acc)

    if valid_acc > best_val_accuracy:
        best_val_accuracy = valid_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"New best model saved with validation accuracy: {valid_acc:.4f}")

    # reduce the learning rate

    lr_scheduler.step()

    pbar.set_postfix_str("Accuracy: Train %.2f%%, Val %.2f%%" % (train_acc * 100, valid_acc * 100))

print("Training complete")

dict = {"training_loss": training_loss_logger,
        "validation_accuracy": validation_acc_logger,
        "training_accuracy": training_acc_logger}

training_logs = pd.DataFrame(dict)
training_logs.to_csv("../results/training_logs.csv", index=False)
