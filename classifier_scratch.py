import torch

from d2l import torch as d2l
from dataset import MNIST

# Simple demonstration of softmax algorithm
# But not very sturdy. Pls use standard one from torch
def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]

@d2l.add_to_class(SoftmaxRegressionScratch)
def forward(self, X):
    X = X.reshape((-1, self.W.shape[0]))
    return softmax(torch.matmul(X, self.W) + self.b)

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()

@d2l.add_to_class(SoftmaxRegressionScratch)
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)

@d2l.add_to_class(d2l.Trainer)  #@save
def __init__(self, max_epochs, num_gpus=0, gradient_clip_val=0):
    self.save_hyperparameters()
    self.gpus = [d2l.gpu(i) for i in range(min(num_gpus, d2l.num_gpus()))]

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    if self.gpus:
        batch = [a.to(self.gpus[0]) for a in batch]
    return batch

@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_model(self, model):
    model.trainer = self
    model.board.xlim = [0, self.max_epochs]
    if self.gpus:
        model.to(self.gpus[0])
    self.model = model

@d2l.add_to_class(d2l.Classifier)
def validation_step(self, batch):
    Y_hat = self(*batch[:-1])
    with open("log.txt", "r+") as fp:
        fp.seek(0, 2)
        fp.write(f"validation_loss: {self.loss(Y_hat, batch[-1])}\n")
        fp.write(f"validation_accuracy: {self.accuracy(Y_hat, batch[-1])}\n")

        fp.close()

    self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
    self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
@d2l.add_to_class(d2l.Module)
def training_step(self, batch):
    l = self.loss(self(*batch[:-1]), batch[-1])
    with open("log.txt", "r+") as fp:
        fp.seek(0, 2)
        fp.write(f"training_loss: {l}\n")
        fp.close()
    self.plot('loss', l, train=True)
    return l


@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    with open("log.txt", "r+") as f:
        f.seek(0, 2)
        f.write(f"epoch {self.epoch}/{self.max_epochs}\n")
        f.close()
    self.model.train()

    print(f"epoch {self.epoch}/{self.max_epochs}\n")
    for batch in self.train_dataloader:
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:  # To be discussed later
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1

#data = MNIST(batch_size=256)

# reset file
fp = open("log.txt", "w")
fp.close()

data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(784, 10, 0.1)
trainer = d2l.Trainer(max_epochs=15)
trainer.fit(model, data)

torch.save(model.state_dict(), 'sofm_reg.params')
d2l.plt.show()
