import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import os
import torch.distributed as dist
from socket import gethostname

from torch.utils.data import DataLoader
import os
import numpy as np
import cupy
import pytorch_lightning as pl


class LightningCNNModel(pl.LightningModule):
    def __init__(self):
        super(LightningCNNModel, self).__init__()

        INPUT_IMAGE_CHANNELS = 3
        INPUT_IMAGE_DIM = 32
        
        CONV_LAYER1_OUTPUT_CHANNELS = 512
        CONV_LAYER1_KERNEL_SIZE = 3
        CONV_LAYER1_STRIDE = 1
        CONV_LAYER1_PADDING = 2
        
        CONV_MAX_POOL_1_KERNEL_SIZE = 2
        CONV_MAX_POOL_1_PADDING_SIZE = 1
        CONV_MAX_POOL_1_STRIDE_SIZE = 1
        
        CONV_LAYER2_OUTPUT_CHANNELS = 1024
        CONV_LAYER2_KERNEL_SIZE = 3
        CONV_LAYER2_STRIDE = 1
        CONV_LAYER2_PADDING = 2
        
        CONV_MAX_POOL_2_KERNEL_SIZE = 2
        CONV_MAX_POOL_2_PADDING_SIZE = 1
        CONV_MAX_POOL_2_STRIDE_SIZE = 2
        
        CONV_LAYER3_OUTPUT_CHANNELS = 256
        CONV_LAYER3_KERNEL_SIZE = 5
        CONV_LAYER3_STRIDE = 2
        CONV_LAYER3_PADDING = 2
        
        CONV_MAX_POOL_3_KERNEL_SIZE = 3
        CONV_MAX_POOL_3_PADDING_SIZE = 0
        CONV_MAX_POOL_3_STRIDE_SIZE = 1
        
        DROP_OUT_RATE = 0.35
        
        LINEAR_LAYER_1_OUTPUT_SIZE = 16834
        LINEAR_LAYER_2_OUTPUT_SIZE = 8192
        
        # NUMBER OF CLASSES
        LINEAR_LAYER_3_OUTPUT_SIZE = 10 # 10 for the amount of classes that are in the dataset

        self.validation_accuracy = []
        self.validation_loss = []

        # CONV2D LAYER1 AND CHANGE IN IMAGE DIMENSIONS
        self.conv_layer1 = nn.Conv2d(INPUT_IMAGE_CHANNELS, CONV_LAYER1_OUTPUT_CHANNELS, CONV_LAYER1_KERNEL_SIZE, CONV_LAYER1_STRIDE, CONV_LAYER1_PADDING)
        self.image_dimension = (INPUT_IMAGE_DIM - ((CONV_LAYER1_KERNEL_SIZE) - (2 * CONV_LAYER1_PADDING)))//CONV_LAYER1_STRIDE + 1
        self.image_channel_size = CONV_LAYER1_OUTPUT_CHANNELS
        

        # MAX POOLING LAYER 1, Change in image dimensions
        self.maxPooling1 = nn.MaxPool2d(CONV_MAX_POOL_1_KERNEL_SIZE, CONV_MAX_POOL_1_STRIDE_SIZE, CONV_MAX_POOL_1_PADDING_SIZE)
        self.image_dimension = (self.image_dimension - ((CONV_MAX_POOL_1_KERNEL_SIZE) - (2 * CONV_MAX_POOL_1_PADDING_SIZE)))//CONV_MAX_POOL_1_STRIDE_SIZE + 1
        
        
        # CONV2D LAYER2 AND CHANGE IN IMAGE DIMENSIONS
        self.conv_layer2 = nn.Conv2d(CONV_LAYER1_OUTPUT_CHANNELS, CONV_LAYER2_OUTPUT_CHANNELS, CONV_LAYER2_KERNEL_SIZE, CONV_LAYER2_STRIDE, CONV_LAYER2_PADDING)
        self.image_dimension = (self.image_dimension - ((CONV_LAYER2_KERNEL_SIZE) - (2 * CONV_LAYER2_PADDING)))//CONV_LAYER2_STRIDE + 1
        self.image_channel_size = CONV_LAYER2_OUTPUT_CHANNELS
        

        # MAX POOLING LAYER 2 AND CHANGE IN IMAGE DIMENSIONS
        self.maxPooling2 = nn.MaxPool2d(CONV_MAX_POOL_2_KERNEL_SIZE, CONV_MAX_POOL_2_STRIDE_SIZE, CONV_MAX_POOL_2_PADDING_SIZE)
        self.image_dimension = (self.image_dimension - ((CONV_MAX_POOL_2_KERNEL_SIZE) - (2 * CONV_MAX_POOL_2_PADDING_SIZE)))//CONV_MAX_POOL_2_STRIDE_SIZE + 1

        
        # CONV2D LAYER 3 AND CHANGE IN IMAGE DIMENSIONS
        self.conv_layer3 = nn.Conv2d(CONV_LAYER2_OUTPUT_CHANNELS, CONV_LAYER3_OUTPUT_CHANNELS, CONV_LAYER3_KERNEL_SIZE, CONV_LAYER3_STRIDE, CONV_LAYER3_PADDING)
        self.image_dimension = (self.image_dimension - ((CONV_LAYER3_KERNEL_SIZE) - (2 * CONV_LAYER3_PADDING)))//CONV_LAYER3_STRIDE + 1
        self.image_channel_size = CONV_LAYER3_OUTPUT_CHANNELS        

        
        # MAX POOLING LAYER 3 AND CHANGE IN IIMAGE DIMENSIONS
        self.maxPooling3 = nn.MaxPool2d(CONV_MAX_POOL_3_KERNEL_SIZE, CONV_MAX_POOL_3_STRIDE_SIZE, CONV_MAX_POOL_3_PADDING_SIZE)
        self.image_dimension = (self.image_dimension - ((CONV_MAX_POOL_3_KERNEL_SIZE) - (2 * CONV_MAX_POOL_3_PADDING_SIZE)))//CONV_MAX_POOL_3_STRIDE_SIZE + 1

        
        # Since we flatten the image after the CONV2D Layers, we need to calculate the size of the feature
        # Vector going into the nn.Linear layer
        self.fc1_input_size = self.image_dimension * self.image_dimension * self.image_channel_size
        
        # Fully connected Layers
        self.fc1 = nn.Linear(self.fc1_input_size, LINEAR_LAYER_1_OUTPUT_SIZE)
        
        self.fc2 = nn.Linear(LINEAR_LAYER_1_OUTPUT_SIZE, LINEAR_LAYER_2_OUTPUT_SIZE)
        
        self.fc3 = nn.Linear(LINEAR_LAYER_2_OUTPUT_SIZE, LINEAR_LAYER_3_OUTPUT_SIZE)

        self.layers = [self.conv_layer1, self.maxPooling1, nn.Dropout(DROP_OUT_RATE), nn.ReLU(),
                       self.conv_layer2, self.maxPooling2, nn.Dropout(DROP_OUT_RATE), nn.ReLU(),
                       self.conv_layer3, self.maxPooling3, nn.Dropout(DROP_OUT_RATE), nn.ReLU(),
                       torch.nn.Flatten(),
                       self.fc1, nn.Dropout(DROP_OUT_RATE), nn.ReLU(),
                       self.fc2, nn.Dropout(DROP_OUT_RATE), nn.ReLU(),
                       self.fc3,
                      ]
        
        self.model_layers = torch.nn.Sequential(*self.layers)
    
    def forward(self, x):
        return self.model_layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = torch.nn.functional.cross_entropy(y_hat, y).item()

        y_hat_np = cupy.asnumpy(y_hat)
        preds = np.argmax(y_hat_np, axis=1)

        self.validation_accuracy.append((preds == y.to('cpu')).sum().item()/len(y))
        self.validation_loss.append(loss)
        return loss
        

    def on_validation_epoch_end(self):
        
        val_acc, val_loss = sum(self.validation_accuracy)/len(self.validation_accuracy), sum(self.validation_loss)/len(self.validation_loss)
        self.validation_accuracy = []
        self.validation_loss = []
        print("Validation Accuracy: ", val_acc, "Validation Loss: ", val_loss)
        print()
        return val_acc, val_loss

    def configure_optimizers(self):
        LEARNING_RATE = 10e-2
        SGD_MOMENTUM = 0.7 # How much of past velocity to maintain in gradient update
        LR_SCHED_GAMMA = 0.85 # Multiplies previous LR by value
    
        # Optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=LEARNING_RATE, momentum=SGD_MOMENTUM, nesterov=True)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=LR_SCHED_GAMMA),
                'interval': 'epoch',
            }
        }



def main():

    BATCH_SIZE = 32
    
    train_kwargs = {'batch_size': BATCH_SIZE}
    test_kwargs = {'batch_size': BATCH_SIZE}
    cuda_kwargs = {'num_workers': 2,
                   'pin_memory': False,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    
    dataset_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download dataset if it is not present on the system
    DOWNLOAD_DATASET = False
    if 'testing' not in os.listdir():
        DOWNLOAD_DATASET = True
        os.mkdir('testing')
    if 'dataset' not in os.listdir('testing'):
        DOWNLOAD_DATASET = True
        os.mkdir('testing/dataset')
        
    if 'training' not in os.listdir():
        DOWNLOAD_DATASET = True
        os.mkdir('training')
    if 'dataset' not in os.listdir('training'):
        DOWNLOAD_DATASET = True
        os.mkdir('training/dataset')
    
    training_data = datasets.CIFAR10(root="training/dataset/", train=True, download=DOWNLOAD_DATASET, transform=dataset_transforms)
    testing_data = datasets.CIFAR10(root="training/dataset/", train=False, download=DOWNLOAD_DATASET, transform=dataset_transforms)

    torch.cuda.empty_cache()

    train_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=BATCH_SIZE,
                                               pin_memory=False)
    
    test_loader = torch.utils.data.DataLoader(testing_data, **test_kwargs)


    ## EDIT THE ENVIRONMENT VARIABLES DEPENDING ON THE NUMBER OF NODES, WORLD SIZE, AND GPUS PER NODE
    # world_size = int(os.environ["WORLD_SIZE"])
    # rank = int(os.environ["SLURM_PROCID"])
    # gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])

    device_count = torch.cuda.device_count()
    
    model = LightningCNNModel()
    trainer = pl.Trainer(max_epochs=15, 
                         devices=device_count, 
                         accelerator="gpu", 
                         strategy='ddp',
                         profiler='advanced',
                         accumulate_grad_batches=3)
    
    trainer.fit(model, train_loader, test_loader)

if __name__ == '__main__':
    main()
