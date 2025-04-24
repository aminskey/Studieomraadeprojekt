import torch
import torchvision
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()
class MNIST(d2l.DataModule):
    def __init__(self, batch_size=64, resize=(28, 28)):
        super().__init__()
        self.save_hyperparameters()

        trans = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        self.train = torchvision.datasets.MNIST(root=self.root, train=True, transform=trans, download=True)
        self.val = torchvision.datasets.MNIST(root=self.root, train=False, transform=trans, download=True)

@d2l.add_to_class(MNIST)
def get_dataloader(self, train):
    data = self.train if train else self.val
    return torch.utils.data.DataLoader(data, self.batch_size, shuffle=train, num_workers=self.num_workers)

@d2l.add_to_class(MNIST)
def text_labels(self, indices):
    return ["{}".format(i) for i in indices]

@d2l.add_to_class(MNIST)
def visualize(self, batch, nrows=3, ncols=8, labels=[]):
    X, y = batch
    if not labels:
        labels = self.text_labels(y)
    d2l.show_images(X.squeeze(1), nrows, ncols, titles=labels)

