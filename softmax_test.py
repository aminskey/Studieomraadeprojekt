"""
* FILE      : softmax_test.py
* AUTHOR    : SYED M. AMIN
* PROJECT   : SOP
* DESC      : Testing the softmax classifier on some handwritten digits from MNIST test dataset
plus 1 image of a hand drawn number taken from the internet
"""

import torch
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from PIL import Image

from classifier_base2 import SoftmaxClassifier, transform, all_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SoftmaxClassifier(784, 10)
model.load_state_dict(torch.load("sofm_cls_scratch.params", weights_only=True, map_location=device))
model = model.to(device)

test_dataset = MNIST("../data", False, transform, download=True)
test_loader = DataLoader(test_dataset, 64, True)

img = Image.open("./test_digit.jpg").convert("L")
img_tensor = all_transforms(img)
img_tensor = img_tensor.to(device)

def plot_test_images(dataloader, num_rows=1, num_cols=10):
    images_shown = 0

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    axes = axes.flatten()
    size = (num_rows * num_cols)

    for images, labels in dataloader:
        for idx in range(size):
            image = images[idx]
            _, pred = torch.max(model(image.to(device)), 1)

            image = image.permute(1, 2, 0)

            axes[idx].imshow(image.cpu().numpy())
            axes[idx].axis("off")
            axes[idx].set_title(f"{pred.item()}")

            images_shown += 1

            if images_shown == size:
                plt.show()
                return

model.eval()

_, pred = torch.max(model(img_tensor), 1)

plt.imshow(img)
plt.axis("off")
plt.title(pred.item())
plot_test_images(test_loader, 5, 5)

