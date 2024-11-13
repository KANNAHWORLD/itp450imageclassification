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
import pytorch_lightning as pl


class LightningCNNModel(pl.LightningModule):
    def __init__(self):
        super(LightningCNNModel, self).__init__()

        self.validation_accuracy = []
        self.validation_loss = []

        INPUT_IMAGE_CHANNELS = 3
        INPUT_IMAGE_DIM = 32
        
        CONV_LAYER1_OUTPUT_CHANNELS = 128
        CONV_LAYER1_KERNEL_SIZE = 3
        CONV_LAYER1_STRIDE = 2
        CONV_LAYER1_PADDING = 2
        
        CONV_MAX_POOL_1_KERNEL_SIZE = 2
        CONV_MAX_POOL_1_PADDING_SIZE = 1
        CONV_MAX_POOL_1_STRIDE_SIZE = 2
        
        CONV_LAYER2_OUTPUT_CHANNELS = 64
        CONV_LAYER2_KERNEL_SIZE = 3
        CONV_LAYER2_STRIDE = 2
        CONV_LAYER2_PADDING = 1
        
        CONV_MAX_POOL_2_KERNEL_SIZE = 2
        CONV_MAX_POOL_2_PADDING_SIZE = 0
        CONV_MAX_POOL_2_STRIDE_SIZE = 2
        
        DROP_OUT_RATE = 0.35
        
        LINEAR_LAYER_1_OUTPUT_SIZE = 512
        LINEAR_LAYER_2_OUTPUT_SIZE = 64
        
        # NUMBER OF CLASSES
        LINEAR_LAYER_3_OUTPUT_SIZE = 10 # 10 for the amount of classes that are in the dataset

        #######################################################################################
        #######################################################################################

        # CONV2D LAYER1 AND CHANGE IN IMAGE DIMENSIONS
        self.conv_layer1 = nn.Conv2d(INPUT_IMAGE_CHANNELS, CONV_LAYER1_OUTPUT_CHANNELS, CONV_LAYER1_KERNEL_SIZE, CONV_LAYER1_STRIDE, CONV_LAYER1_PADDING)
        self.image_dimension = (INPUT_IMAGE_DIM - ((CONV_LAYER1_KERNEL_SIZE) - (2 * CONV_LAYER1_PADDING)))//CONV_LAYER1_STRIDE + 1
        self.image_channel_size = CONV_LAYER1_OUTPUT_CHANNELS
        print(self.image_dimension, self.image_channel_size)


        # MAX POOLING LAYER 1, Change in image dimensions
        self.maxPooling1 = nn.MaxPool2d(CONV_MAX_POOL_1_KERNEL_SIZE, CONV_MAX_POOL_1_STRIDE_SIZE, CONV_MAX_POOL_1_PADDING_SIZE)
        self.image_dimension = (self.image_dimension - ((CONV_MAX_POOL_1_KERNEL_SIZE) - (2 * CONV_MAX_POOL_1_PADDING_SIZE)))//CONV_MAX_POOL_1_STRIDE_SIZE + 1
        print(self.image_dimension, self.image_channel_size)

        
        # CONV2D LAYER2 AND CHANGE IN IMAGE DIMENSIONS
        self.conv_layer2 = nn.Conv2d(CONV_LAYER1_OUTPUT_CHANNELS, CONV_LAYER2_OUTPUT_CHANNELS, CONV_LAYER2_KERNEL_SIZE, CONV_LAYER2_STRIDE, CONV_LAYER2_PADDING)
        self.image_dimension = (self.image_dimension - ((CONV_LAYER2_KERNEL_SIZE) - (2 * CONV_LAYER2_PADDING)))//CONV_LAYER2_STRIDE + 1
        self.image_channel_size = CONV_LAYER2_OUTPUT_CHANNELS
        print(self.image_dimension, self.image_channel_size)


        # MAX POOLING LAYER 2 AND CHANGE IN IMAGE DIMENSIONS
        self.maxPooling2 = nn.MaxPool2d(CONV_MAX_POOL_2_KERNEL_SIZE, CONV_MAX_POOL_2_STRIDE_SIZE, CONV_MAX_POOL_2_PADDING_SIZE)
        self.image_dimension = (self.image_dimension - ((CONV_MAX_POOL_2_KERNEL_SIZE) - (2 * CONV_MAX_POOL_2_PADDING_SIZE)))//CONV_MAX_POOL_2_STRIDE_SIZE + 1
        print(self.image_dimension, self.image_channel_size)


        # Since we flatten the image after the CONV2D Layers, we need to calculate the size of the feature
        # Vector going into the nn.Linear layer
        self.fc1_input_size = self.image_dimension * self.image_dimension * self.image_channel_size
        
        # Fully connected Layers
        self.fc1 = nn.Linear(self.fc1_input_size, LINEAR_LAYER_1_OUTPUT_SIZE)
        print(self.fc1_input_size, LINEAR_LAYER_1_OUTPUT_SIZE)
        
        self.fc2 = nn.Linear(LINEAR_LAYER_1_OUTPUT_SIZE, LINEAR_LAYER_2_OUTPUT_SIZE)
        print(LINEAR_LAYER_1_OUTPUT_SIZE, LINEAR_LAYER_2_OUTPUT_SIZE)

        
        self.fc3 = nn.Linear(LINEAR_LAYER_2_OUTPUT_SIZE, LINEAR_LAYER_3_OUTPUT_SIZE)
        print(LINEAR_LAYER_2_OUTPUT_SIZE, LINEAR_LAYER_3_OUTPUT_SIZE)


        self.layers = [self.conv_layer1, self.maxPooling1, nn.Dropout(DROP_OUT_RATE), nn.ReLU(),
                       self.conv_layer2, self.maxPooling2, nn.Dropout(DROP_OUT_RATE), nn.ReLU(),
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
        preds = np.argmax(y_hat.to('cpu'), axis=1)

        y = y.to('cpu')
        self.validation_accuracy.append((preds == y).sum().item()/len(y))
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
        LEARNING_RATE = 10e-3
        SGD_MOMENTUM = 0.7 # How much of past velocity to maintain in gradient update
        LR_SCHED_GAMMA = 0.65 # Multiplies previous LR by value
    
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
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': int(os.environ["SLURM_CPUS_PER_TASK"]),
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


    train_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=args.batch_size,
                                               pin_memory=False)

    
    test_loader = torch.utils.data.DataLoader(testing_data,
                                              **test_kwargs)

    #######################################################################################
    #######################################################################################
    
    

    #######################################################################################
    #######################################################################################
   
   
    model = LightningCNNModel()
    trainer = pl.Trainer(max_epochs=30, num_nodes=1)
    trainer.fit(model, train_loader, test_loader)

    


if __name__ == '__main__':
    main()


