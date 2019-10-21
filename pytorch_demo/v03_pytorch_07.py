"""
用pytorch替代numpy实现两层神经网络
"""
import torch
import torch.nn as nn

N, D_in, H, D_out = 64, 100, 100, 10
# 随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

class TwoLayerNet(torch.nn.Module):
    def __init__(self,D_in,H,D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 =  torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x) :
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred

model = TwoLayerNet(D_in,H,D_out)

loss_fn = nn.MSELoss(reduction='sum')
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for it in range(500):
    # Forward pass
    y_pred = model(x)

    # compute loss
    loss = loss_fn(y_pred,y)
    print(it, loss.item())

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # update weights
    optimizer.step()