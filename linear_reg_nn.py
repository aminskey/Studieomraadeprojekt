import torch
from torch import nn
from d2l import torch as d2l

class LinearRegression(d2l.Module):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()

        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)

@d2l.add_to_class(LinearRegression)
def forward(self, X):
    return self.net(X)

@d2l.add_to_class(LinearRegression)
def loss(self, y_hat, y):
    fn = nn.MSELoss()
    return fn(y_hat, y)

@d2l.add_to_class(LinearRegression)
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), self.lr)

@d2l.add_to_class(LinearRegression)
def get_w_b(self):
    return (self.net.weight.data, self.net.bias.data)

if __name__ == '__main__':
    model = LinearRegression(lr=0.03)
    data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2)
    trainer = d2l.Trainer(max_epochs=3)
    trainer.fit(model, data)

    w, b = model.get_w_b()

    print(f'error in estimating w: {data.w - w.reshape(data.w.shape)}')
    print(f'error in estimating b: {data.b - b}')


    d2l.plt.show()
