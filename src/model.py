# Reproduction of Paper

# Create the model used for the results of fig. 6 trained on RetinaMNIST
# Fully connected Neural Net with dimensions (d-512-K)
# with d=(3x28x28), K=5, and batch size M=8

# softmax at the final layer? is applied in training mode, not as layer

# Setup
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from src.network import TorchNet

# Requirements for RetinaMNIST dataset
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator

# Training
from tqdm import tqdm


# run on GPU if available
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


def test(model, split, BATCH_SIZE, task, train_dataset, test_dataset, data_flag):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                                           shuffle=False)  # CAG: changed "2*BATCH_SIZE" to "BATCH_SIZE"
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                                  shuffle=False)  # CAG: changed "2*BATCH_SIZE" to "BATCH_SIZE"

    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.view(BATCH_SIZE, -1).to(device)  # CAG: is it okay to flatten?
            outputs = model(inputs).to(device)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32).to(device)
                outputs = outputs.softmax(dim=-1).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                outputs = outputs.softmax(dim=-1).to(device)
                targets = targets.float().resize_(len(targets), 1).to(device)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()

        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)

        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))


def init_model(NUM_EPOCHS):
    print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

    # Loader RetinaMNIST dataset
    data_flag = 'retinamnist'
    download = True

    BATCH_SIZE = 40
    lr = 0.001

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

    # print(train_dataset)
    # print("===================")
    # print(test_dataset)

    # Montage
    train_dataset.montage(length=20)

    # check sizes of dataset and train_loader
    x, y = train_dataset[0]

    print("Dataset and train_loader sizes:")
    print(x.shape, y.shape)

    for x, y in train_loader:
        print(x.shape, y.shape)
        break

    # Initialize Pytorch network
    in_features = 3 * 28 * 28
    hidden_dim = 512
    out_features = 5

    model = TorchNet(in_features, hidden_dim, out_features)
    # print(model)

    # define loss function and optimizer
    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        criterion_vec = nn.CrossEntropyLoss(reduction='none')

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # train
    losses = []

    for epoch in tqdm(range(NUM_EPOCHS)):
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0
        total_loss = 0

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
            total_loss += loss.cpu().detach().numpy()
            optimizer.step()

        losses.append(total_loss)

    # Plot loss against epochs
    plt.plot(range(NUM_EPOCHS), losses)
    plt.show()

    # Get inputs, gradients, and weights
    x = inputs
    y = targets
    dLdy = loss_vec
    dLdw2 = model.layer2.weight.grad
    dLdb2 = model.layer2.bias.grad
    dLdw1 = model.layer1.weight.grad
    dLdb1 = model.layer1.bias.grad
    gradients = [dLdw1, dLdb1, dLdw2, dLdb2]

    print("Size of dLdy", dLdy.shape, "should be equal to", y.shape)

    print('==> Evaluating ...')
    test(model, 'train', BATCH_SIZE, task, train_dataset, test_dataset, data_flag)
    test(model, 'test', BATCH_SIZE, task, train_dataset, test_dataset, data_flag)

    return model, gradients
