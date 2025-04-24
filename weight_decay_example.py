import torch
from torch import nn
from d2l import torch as d2l

class WeightDecay(d2l.LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.wd = wd

    def configure_optimizers(self):
        return torch.optim.SGD([
            {'params': self.net.weight, 'weight_decay': self.wd},
            {'params': self.net.bias}
        ], lr=self.lr)

class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = torch.randn(n, num_inputs)
        noise = torch.randn(n, 1) * 0.01
        w, b = torch.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = torch.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)

def l2_penalty(w):
    return (w**2).sum()/2

if __name__ == '__main__':
    model = WeightDecay(wd=3, lr=0.01)
    data = Data(20, 100, 200, 5)
    trainer = d2l.Trainer(max_epochs=50)

    model.board.yscale = 'log'
    trainer.fit(model, data)

    print('L2 norm of w:', float(l2_penalty(model.get_w_b()[0])))
    d2l.plt.show()