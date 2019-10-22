"""
游戏
"""
import numpy as np
import torch
import time
import datetime

USE_CUDA = torch.cuda.is_available()
# USE_CUDA = False
NUM_DIGITS = 10
# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0


def fizz_buzz_decode(i, prediction):
    return [str(i+1), "fizz", "buzz", "fizzbuzz"][prediction]

# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

trX = torch.Tensor([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = torch.LongTensor([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

# Define the model
NUM_HIDDEN = 1000
model = torch.nn.Sequential(
    torch.nn.Linear(NUM_DIGITS, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, 4)
)
if USE_CUDA:
    model = model.cuda()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.05)

# start = time.clock()
starttime = datetime.datetime.now()
# Start training it
BATCH_SIZE = 512
for epoch in range(10000):
    for start in range(0, len(trX), BATCH_SIZE):
        end = start + BATCH_SIZE
        batchX = trX[start:end]
        batchY = trY[start:end]

        if USE_CUDA:
            batchX = batchX.cuda()
            batchY = batchY.cuda()

        y_pred = model(batchX)
        loss = loss_fn(y_pred, batchY)
        print("Epoch",epoch,loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# elapsed = (time.clock() - start)
# print("Time used:",elapsed)
endtime = datetime.datetime.now()

seconds = (endtime - starttime).seconds
start = starttime.strftime('%Y-%m-%d %H:%M')
# 100 秒
# 分钟
minutes = seconds // 60
second = seconds % 60
print((endtime - starttime))
timeStr = str(minutes) + '分钟' + str(second) + "秒"
print("程序从 " + start + ' 开始运行,运行时间为：' + timeStr)

testX = torch.Tensor([binary_encode(i,NUM_DIGITS)for i in range(1,101)])
if USE_CUDA:
    testX = testX.cuda()
with torch.no_grad():
    testY = model(testX)

predictions = zip(range(0,101),testY.max(1)[1].cpu().data.tolist())
print([fizz_buzz_decode(i,x)for i,x in predictions])