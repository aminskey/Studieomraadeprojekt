"""
* FILE      : classifier_base2.py
* AUTHOR    : SYED M. AMIN
* PROJECT   : SOP
* DESC      : a file containing all the softmax classifier class and some transforms
for image recognition
"""


import torch

from torchvision import transforms

class SoftmaxClassifier(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.linear = torch.nn.Linear(n_inputs, n_outputs)
        self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)

# preproccessing template
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

all_transforms = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,),
                         std=(0.5,))
])