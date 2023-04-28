import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.network import TorchNet
from src.algo1 import algo1
from src.algo3 import algo3
from src.constants import *

import medmnist
from medmnist import INFO, Evaluator
import torchvision.transforms as transforms
import torch.utils.data as data


# run on GPU if available
if torch.cuda.is_available():
    print("Using GPU")
    dev = "cuda:0"
else:
    print("Using CPU")
    dev = "cpu"
device = torch.device(dev)


def find_insecure_batch(NUM_EPOCHS):
    data_flag = 'retinamnist'
    download = True

    info = INFO[data_flag]
    task = info['task']

    n_channels = info['n_channels']
    n_classes = len(info['label'])

    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])  # correct for the retina dataset?

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    # pil_dataset = DataClass(split='train', download=download)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Creat for loop to find insecure batch
    for i in range(5000):
        print(f"Starting itteration {i}")
        # Initialize Pytorch network
        model = TorchNet(in_features, hidden_dim, out_features)

        # define loss function and optimizer
        if task == "multi-label, binary-class":
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
            criterion_vec = nn.CrossEntropyLoss(reduction='none')

        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        # Train
        for epoch in tqdm(range(NUM_EPOCHS)):
            train_correct = 0
            train_total = 0
            test_correct = 0
            test_total = 0

            test_loss = 0

            model.train()
            for inputs, targets in train_loader:
                # forward + backward + optimize
                optimizer.zero_grad()
                inputs = inputs.view(BATCH_SIZE, -1).to(device)  # CAG: is it okay to flatten?
                outputs = model(inputs)
                targets = targets.to(device)

                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    loss = criterion(outputs, targets)
                else:
                    targets = targets.squeeze().long()
                    loss = criterion(outputs, targets)
                    loss_vec = criterion_vec(outputs, targets)

                loss.backward()
                optimizer.step()

        # Retrieve relevant info
        x = inputs
        y = targets
        loss = loss_vec
        dLdW2 = model.layer2.weight.grad
        dLdb2 = model.layer2.bias.grad
        dLdW1 = model.layer1.weight.grad
        dLdb1 = model.layer1.bias.grad
        gradients = [dLdW1, dLdb1, dLdW2, dLdb2]

        g_mc, I = algo1(gradients)
        Dmh = algo3(gradients, I, g_mc)
