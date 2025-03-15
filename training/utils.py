import torch 
from datasets import load_dataset
from tqdm import tqdm
from datasets import Dataset
import torch 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import torchvision


def extract_patches(image_tensor, patch_size = 4):
    bs, c, h, w = image_tensor.size()

    # define teh unfold layer with appropriate parameters 

    unfold = torch.nn.Unfold(kernel_size = patch_size, stride = patch_size)

    unfolded = unfold(image_tensor)

    # reshape the unfolded tensor to match the desired output shape 
    # output shaep BS x L x C x 8 x8 where L is the number of patches in each dimension 
    # fo reach dimension, number of patches = (original dimension size) //patch_size 

    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)

    return unfolded


# we have a hugging face dataset, so we now define a custom dataset class which processes this, ensures our labels are one hot encoded etc. 

class MultiLabelDataset(Dataset):
    """
    Initialize with a Hugging Face dataset that's already formatted as torch tensors. 
    Will convert to tensors if plain hugging face dataset. 
    Will handle the multi label nature of our data through one hot encoding 
    
    Args:
        hf_dataset: A Hugging Face dataset with 'image' and 'labels' columns
    """
      
    def __init__(self, hf_dataset, image_size):

        self.x_train, self.x_val, self.y_train, self.y_val = None, None, None, None
        self.mode = "train"

        hf_dataset = hf_dataset.with_format("torch")
        print(hf_dataset.format)

        self.processed_images = []
        self.processed_labels = []

        for sample in tqdm(hf_dataset, desc= "processing image files"):
            image = sample['image']

            # we resize our image if specified 
            
            image = F.interpolate(image.unsqueeze(0), size = image_size, mode = "bilinear").squeeze(0)


            if image.shape[0] == 4:
                image = torch.index_select(image, 0, torch.tensor([0]))

            # normalize pixel values 
            image = image/255

            if image.shape[0] == 1:
                image = image.repeat(3, 1, 1) # for convolutional networks we need 3 channels, remove this line and line below if wanting 1024, 1024 shape 

            # image = image.permute(1, 2, 0) # channel dimension needs to be the last one, not first 

            labels = sample['labels']
            one_hot = torch.zeros(15, dtype = torch.long)
            one_hot[labels] = 1
            self.processed_images.append(image)
            self.processed_labels.append(one_hot)
    
    def train_validation_split(self):
        """
        Takes our training data and produces a validation set from our training data
        Ensures that we don't use our hugging face defined test set during the training process 
        Means that we will assess trained models on a totally separate test set to avoid overfitting on test set
        Note that we use a subset of train as a validation set 
        """
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.processed_images, self.processed_labels, test_size = 0.2,
                                                                              random_state = 42)
        

    def __len__(self):
        """ Returns the length of our training or validation set depending on mode """
        if self.mode == "train":
            return len(self.x_train)
        elif self.mode == "val":
            return len(self.x_val)
        elif self.mode == "test":
            return len(self.processed_images)
    
        

    # note we are not doing lazy processing, so our data is processed when the dataset is instantiated. 
    # here we will return a train test split of our hugging face training data, unless we're using the test data in which case we return it all 
    def __getitem__(self, idx):
        """Gets items from either the training or validation set depending on mode"""
        if self.mode == "train":
            return {"image": self.x_train[idx], "labels": self.y_train[idx]}
        elif self.mode == "val":
            return {"image": self.x_val[idx], "labels": self.y_val[idx]}
        elif self.mode == "test":
            return {"image": self.processed_images[idx], "labels": self.processed_labels[idx]}
    
        



# we create patches 
# we embed these patches by using encoder only transformer 
# then we pass them through an encoder only transformer 
# use a transformer - self attention mixes spatial regions of an image earlier on (rather than convolutions which takes time to get entire spatial field of image)
# by treating every pixel as an embedding in a sequence, each spatial region can query all other spatial regions in image - gives context 

# neeed to start by writing a transformer block class with self attention 

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size = 128, num_heads = 4):
        super(TransformerBlock, self).__init__()


        # layer normalisation to normalize the input data 
        self.norm1 = nn.LayerNorm(hidden_size)


        # multi head attention mechanism 

        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads, 
                                                    batch_first = True, dropout = 0.1)
        
        # another layer of normalisation 

        self.norm2 = nn.LayerNorm(hidden_size)

        # multi layer perceptron with a hidden layer and activation function 

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size *2), 
            nn.LayerNorm(hidden_size * 2),
            nn.ELU(), 
            nn.Linear(hidden_size * 2, hidden_size))
        
    def forward(self, x):

        # apply the first layer of normalisation 

        norm_x = self.norm1(x)

        # apply multi headed attention and add the input (residual connection)

        x = self.multihead_attn(norm_x, norm_x, norm_x)[0] + x

        # apply second layer of normalisation 

        norm_x = self.norm2(x)

        # pass through the mlp and add the input (Residual connection)

        x = self.mlp(norm_x) + x

        return x 
        

class ViT(nn.Module):
    def __init__(self, image_size, channels_in, patch_size, hidden_size,
                 num_layers, num_heads = 8):
        super(ViT, self).__init__()

        self.patch_size = patch_size

        # fully connected layer to project input patches to the hidden size dimension 

        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size) # this is causing a problem atm 

        # create list of transformer blocks 

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads) for _ in range(num_layers)])
        
        # fully connected output layer to map to the number of classes 

        self.fc_out = nn.Linear(hidden_size, 15)

        # parameter for the output token 

        self.out_vec = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # positional embeddings to retain positional information of patches 

        seq_length = (image_size //patch_size) **2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_size).normal_(std = 0.001))

    def forward(self, image):

        bs = image.shape[0]

        # extract patches from the image and flatten them 

        patch_seq = extract_patches(image, patch_size = self.patch_size)

        

        # project patches to the hidden size dimension 

        patch_emb = self.fc_in(patch_seq)


        # add positional embeddings to the patch embeddings 

        patch_emb = patch_emb + self.pos_embedding

        # concatenate the output token to the patch embeddings 

        embs = torch.cat((self.out_vec.expand(bs, 1, -1), patch_emb), 1)

        # pass embeddings through each transformer block 

        for block in self.blocks: 
            embs = block(embs)

        return self.fc_out(embs[:, 0])


