"""
用pytorch简化
"""
import torch
import torch.nn as nn

N, D_in, H, D_out = 64, 100, 100, 10
# 随机创建一些训练数据
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)
# model = torch.nn.Sequential(
#     torch.nn.Linear(D_in, H),
#     torch.nn.ReLU(),
#     torch.nn.Linear(H, D_out)
# )
# # model = model.cuda()

loss_fn = nn.MSELoss(reduction='sum')

learning_rate = 1e-4

for it in range(500):
    # Forward pass
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    # y_pred = model(x)

    # compute loss
    loss = (y_pred - y).pow(2).sum()
    # loss = loss_fn(y_pred,y)
    print(it, loss.item())

    # Backward pass
    # model.zero_grad()
    loss.backward()

    # update weights of w1 & w2
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        w1.grad.zero_()
        w2.grad.zero_()
        # for param in model.parameters():
        #     param -= learning_rate * param.grad
