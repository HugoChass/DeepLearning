# Reproduction of Paper

# Create the model used for the results of fig. 6 trained on RetinaMNIST
# Fully connected Neural Net with dimensions (d-512-K)
# with d=(3x28x28), K=5, and batch size M=8

# softmax at the final layer? is applied in training mode, not as layer

# Setup
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Requirements for RetinaMNIST dataset
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO, Evaluator

# Training
from tqdm import tqdm

print(f"MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE}")

# Loader RetinaMNIST dataset
data_flag = 'retinamnist'
download = True

NUM_EPOCHS = 1
BATCH_SIZE = 8
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

pil_dataset = DataClass(split='train', download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                                       shuffle=False)  # CAG: changed "2*BATCH_SIZE" to "BATCH_SIZE"
test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                              shuffle=False)  # CAG: changed "2*BATCH_SIZE" to "BATCH_SIZE"

print(train_dataset)
print("===================")
print(test_dataset)

# Montage
train_dataset.montage(length=20)

# check sizes of dataset and train_loader
x, y = train_dataset[0]

print(x.shape, y.shape)

for x, y in train_loader:
    print(x.shape, y.shape)
    break


class TorchNet(nn.Module):
    """
    PyTorch neural network. Network layers are defined in __init__ and forward
    pass implemented in forward.
    
    Args:
        in_features: number of features in input layer
        hidden_dim: number of features in hidden dimension
        out_features: number of features in output layer
    """

    def __init__(self, in_features, hidden_dim, out_features):
        super(TorchNet, self).__init__()

        self.layer1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, out_features)

    def forward(self, x):
        # x = torch.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# Initialize Pytorch network
in_features = 3 * 28 * 28
hidden_dim = 512
out_features = 5

model = TorchNet(in_features, hidden_dim, out_features)
print(model)

# define loss function and optimizer
if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()
    criterion_vec = nn.CrossEntropyLoss(reduction='none')

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# train

for epoch in range(NUM_EPOCHS):
    train_correct = 0
    train_total = 0
    test_correct = 0
    test_total = 0

    model.train()
    for inputs, targets in tqdm(train_loader):
        # forward + backward + optimize
        optimizer.zero_grad()
        inputs = inputs.view(BATCH_SIZE, -1)  # CAG: is it okay to flatten?
        outputs = model(inputs)

        if task == 'multi-label, binary-class':
            targets = targets.to(torch.float32)
            loss = criterion(outputs, targets)
        else:
            targets = targets.squeeze().long()
            loss = criterion(outputs, targets)
            loss_vec = criterion_vec(outputs, targets)

        loss.backward()
        optimizer.step()

# Get inputs, gradients, and weights
x = inputs
y = targets
dLdy = loss_vec
dLdw2 = model.layer2.weight.grad
dLdb2 = model.layer2.bias.grad
dLdw1 = model.layer1.weight.grad
dLdb1 = model.layer1.bias.grad

print("Size of dLdy", dLdy.shape, "should be equal to", y.shape)


# evaluation

def test(split):
    model.eval()
    y_true = torch.tensor([])
    y_score = torch.tensor([])

    data_loader = train_loader_at_eval if split == 'train' else test_loader

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.view(BATCH_SIZE, -1)  # CAG: is it okay to flatten?
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
            else:
                targets = targets.squeeze().long()
                outputs = outputs.softmax(dim=-1)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.numpy()
        y_score = y_score.detach().numpy()

        evaluator = Evaluator(data_flag, split)
        metrics = evaluator.evaluate(y_score)

        print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))


print('==> Evaluating ...')
test('train')
test('test')

# 1080 samples in training set, when batch size of 8 is used, it takes 135 forward and backward passes to calculate results and gradients
# When the batch size is set to twice as much for evaluating, 1080/16 = 67.5 the last input to evaluate will have half the size which will result in an error
# Since speed is not as important, the batch size for evaluating has been set to same as training: 8
# Idendito for the test loader because of the last line: test('test')
# H is the amount of hidden layers
