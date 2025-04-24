"""
* FILE      : classifier_scratch2.py
* AUTHOR    : SYED M. AMIN
* PROJECT   : SOP
* DESC      : the program that trains the softmax classifier and saves it
"""


import torch
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from classifier_base2 import SoftmaxClassifier, transform

LR = 1e-3
num_classes = 10
num_epochs = 10
batch_size = 64

"""
model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(28*28, 10)
)"""


model = SoftmaxClassifier(784, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lossfunc = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=LR)

# loading dataset
train_dataset = MNIST(root='../data', train=True, transform=transform, download=True)

# preparing sizes
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

# plots list
loss_plots = []
acc_plots = []
val_loss_plots = []

def eval_model(loader):
    model.eval()
    total_loss = 0
    total_corr = 0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            images, labels = batch
            outputs = model(images.to(device))
            loss = lossfunc(outputs, labels.to(device))
            _, predicted = torch.max(outputs, 1)  # taking the most accurate prediction .. max(P(X = x))
            correct = (predicted == labels.to(device)).sum().item()

            total_loss += loss
            total_corr += correct

            total_samples += batch[1].size(0)

    # calculating average loss and accuracy based on sample size
    avg_loss = total_loss / total_samples
    accuracy = total_corr / total_samples

    return avg_loss, accuracy
def model_fit(epochs):
    for epoch in range(epochs):
        total_loss = 0
        total_acc = 0

        print(f"epoch {epoch+1}/{epochs}")

        batch_idx = 0
        epoch_loss = 0
        for images, labels in train_loader:
            outputs = model(images.to(device))             # prediction
            loss = lossfunc(outputs, labels.to(device))    # calculating loss

            optim.zero_grad()           # resetting gradient history
            loss.backward()             # calculating the gradient
            optim.step()                # implementing gradient descent

            total_loss += loss.item()   # Getting the value of the loss instead of entire tensor
            epoch_loss += loss.item()
            batch_idx += 1

        epoch_loss /= len(train_loader)
        print(f"epoch nr. {epoch+1}: training_loss: {epoch_loss}")
        loss_plots.append(epoch_loss)

        model.eval()
        l, acc = eval_model(val_loader)

        print(f"epoch nr. {epoch+1}: validation_loss: {l}")
        print(f"epoch nr. {epoch+1}: validation_acc: {acc}")

        acc_plots.append(acc)
        val_loss_plots.append(l)

model_fit(num_epochs)
torch.save(model.state_dict(), 'sofm_cls_scratch.params')

plt.plot(range(0, len(loss_plots)), loss_plots, marker='.', linestyle='-', label='Training Loss')
plt.plot(range(0, len(acc_plots)), acc_plots, marker='.', linestyle='-', label='Validation Accuracy')
plt.plot(range(0, len(val_loss_plots)), val_loss_plots, marker='.', linestyle='-', label='Validation Loss')

plt.xlabel('Epoch')
plt.ylabel('Metrics')
plt.title('Training and Validation Metrics')
plt.legend()
plt.grid()
plt.show()