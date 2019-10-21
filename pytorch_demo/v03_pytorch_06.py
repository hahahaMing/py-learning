"""
用optimizer简化
"""
import torch
import torch.nn as nn

N, D_in, H, D_out = 64, 100, 100, 10
# 随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)
# torch.nn.init.normal_(model[0].weight)
# torch.nn.init.normal_(model[2].weight)
# model = model.cuda()

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