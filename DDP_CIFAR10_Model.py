import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname

import tqdm

import cudf
import cupy
from torch.utils.data import DataLoader
import os

class serialized_CNN_Model(nn.Module):
    def __init__(self):
        super(serialized_CNN_Model, self).__init__()

        INPUT_IMAGE_CHANNELS = 3
        INPUT_IMAGE_DIM = 32
        
        CONV_LAYER1_OUTPUT_CHANNELS = 256
        CONV_LAYER1_KERNEL_SIZE = 3
        CONV_LAYER1_STRIDE = 1
        CONV_LAYER1_PADDING = 0
        
        CONV_MAX_POOL_1_KERNEL_SIZE = 2
        CONV_MAX_POOL_1_PADDING_SIZE = 0
        CONV_MAX_POOL_1_STRIDE_SIZE = 2
        
        CONV_LAYER2_OUTPUT_CHANNELS = 1024
        CONV_LAYER2_KERNEL_SIZE = 3
        CONV_LAYER2_STRIDE = 1
        CONV_LAYER2_PADDING = 2
        
        CONV_MAX_POOL_2_KERNEL_SIZE = 2
        CONV_MAX_POOL_2_PADDING_SIZE = 0
        CONV_MAX_POOL_2_STRIDE_SIZE = 2
        
        CONV_LAYER3_OUTPUT_CHANNELS = 256
        CONV_LAYER3_KERNEL_SIZE = 5
        CONV_LAYER3_STRIDE = 2
        CONV_LAYER3_PADDING = 0
        
        CONV_MAX_POOL_3_KERNEL_SIZE = 3
        CONV_MAX_POOL_3_PADDING_SIZE = 0
        CONV_MAX_POOL_3_STRIDE_SIZE = 2
        
        DROP_OUT_RATE = 0.35
        
        LINEAR_LAYER_1_OUTPUT_SIZE = 8192
        LINEAR_LAYER_2_OUTPUT_SIZE = 8192
        
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

        
        # CONV2D LAYER 3 AND CHANGE IN IMAGE DIMENSIONS
        self.conv_layer3 = nn.Conv2d(CONV_LAYER2_OUTPUT_CHANNELS, CONV_LAYER3_OUTPUT_CHANNELS, CONV_LAYER3_KERNEL_SIZE, CONV_LAYER3_STRIDE, CONV_LAYER3_PADDING)
        self.image_dimension = (self.image_dimension - ((CONV_LAYER3_KERNEL_SIZE) - (2 * CONV_LAYER3_PADDING)))//CONV_LAYER3_STRIDE + 1
        self.image_channel_size = CONV_LAYER3_OUTPUT_CHANNELS
        print(self.image_dimension, self.image_channel_size)
        

        # MAX POOLING LAYER 3 AND CHANGE IN IIMAGE DIMENSIONS
        self.maxPooling3 = nn.MaxPool2d(CONV_MAX_POOL_3_KERNEL_SIZE, CONV_MAX_POOL_3_STRIDE_SIZE, CONV_MAX_POOL_3_PADDING_SIZE)
        self.image_dimension = (self.image_dimension - ((CONV_MAX_POOL_3_KERNEL_SIZE) - (2 * CONV_MAX_POOL_3_PADDING_SIZE)))//CONV_MAX_POOL_3_STRIDE_SIZE + 1
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
                       self.conv_layer3, self.maxPooling3, nn.Dropout(DROP_OUT_RATE), nn.ReLU(),
                       torch.nn.Flatten(),
                       self.fc1, nn.Dropout(DROP_OUT_RATE), nn.ReLU(),
                       self.fc2, nn.Dropout(DROP_OUT_RATE), nn.ReLU(),
                       self.fc3,
                      ]
        
        self.model_layers = [torch.nn.Sequential(*self.layers)]

    def forward(self, x):
        for i in range(len(self.model_layers)):
            x = self.model_layers[i](x)

        return x




def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        

def test(model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output= model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss

            predictions = cupy.argmax(output, axis=1)
            correct += (predictions == target).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    print('\n Test set avrage loss {:.4f}'.format(test_loss))

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

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

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    
    print(f"Hello from rank {rank} of {world_size} on {gethostname()} where there are" \
          f" {gpus_per_node} allocated GPUs per node.", flush=True)

    setup(rank, world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    train_sampler = torch.utils.data.distributed.DistributedSampler(training_data,
                                                                    num_replicas=world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(training_data,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]),
                                               pin_memory=True)

    
    test_loader = torch.utils.data.DataLoader(testing_data,
                                              **test_kwargs)

    #######################################################################################
    #######################################################################################
    
    # Dataset Constants
    DATASET_SIZE = 50000
    
    # Training Hyperparams
    BATCH_SIZE = 5
    EPOCHS = 30
    
    # Optimizer Hyperparams
    LEARNING_RATE = 10e-3
    SGD_MOMENTUM = 0.9 # How much of past velocity to maintain in gradient update
    
    # LR Scheduler Hyperparams
    GAMMA = 0.95 # Multiplies previous LR by 0.1
    STEP_SIZE = 50000//BATCH_SIZE # Amount of steps before LR is decreased

    #######################################################################################
    #######################################################################################
    
    model = serialized_CNN_Model().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=SGD_MOMENTUM, nesterov=True)
    
    # LR Scheduler
    learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    
    # Loss func
    criterion = nn.CrossEntropyLoss()
    print("Initial testing before model training")
    test(ddp_model, local_rank, criterion, test_loader)
    for epoch in range(1, args.epochs + 1):
        train(args, ddp_model, local_rank, train_loader, optimizer, criterion, epoch)
        torch.cuda.empty_cache()
        if rank == 0: test(ddp_model, local_rank, criterion, test_loader)
        learning_rate_scheduler.step()

    # if args.save_model and rank == 0:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
