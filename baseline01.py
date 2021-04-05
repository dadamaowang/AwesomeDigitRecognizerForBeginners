# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable  # Variable是一种可以自己求导和梯度下降的变量
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
# train_test_split帮助我们从训练集中分出验证集

# prepare dataset

# load data
train_set = pd.read_csv('../drdata/train.csv', dtype=np.float32)


# split data into features and labels
labels_np = train_set.label.values
images_np = train_set.loc[:, train_set.columns != 'label'].values / 255  # 除以255,normalization
train_images, val_images, train_labels, val_labels = \
    train_test_split(images_np, labels_np, test_size=0.2, random_state=42)  # 20%划分出去作为验证集
# train_test_split  inputs: (samples, labels)
# outputs: train_sample, test_sample, train_label, test_label

# create tensor to create Variables that could accumulate gradients
TrainImages = torch.from_numpy(train_images)
TrainLabels = torch.from_numpy(train_labels).type(torch.LongTensor)
ValImages = torch.from_numpy(val_images)
ValLabels = torch.from_numpy(val_labels).type(torch.LongTensor)

# 33600 samples to train
batch_size = 100
n_iters = 336
num_epoches = 100

# pytorch train and test datasets
train = torch.utils.data.TensorDataset(TrainImages, TrainLabels)
val = torch.utils.data.TensorDataset(ValImages, ValLabels)

# data loader
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)


# Create Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()

        # Linear Part
        self.linear = nn.Linear(input_dim, output_dim)
        # 还需要一个logistic function, 但是在Pytorch中它被包含在loss里
        # 因此模型框架搭建的时候，logistic fuction是不必写的

    def forward(self, x):
        out = self.linear(x)
        return out


input_dim = 28*28
output_dim = 10
model = LogisticRegressionModel(input_dim, output_dim)
error = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training the model
epoch_list = []
loss_list = []
for epoch in range(num_epoches):
    for i, (images, labels) in enumerate(train_loader):
        train = Variable(images.view(-1, 28*28))
        labels = Variable(labels)

        optimizer.zero_grad()
        outputs = model(train)
        loss = error(outputs, labels)
        loss.backward()
        optimizer.step()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            test = Variable(images.view(-1, 28*28))
            outputs = model(test)
            predicted = torch.max(outputs.data, 1)[1]
            # 输出的是每个样本的一个分类概率矩阵，取其中最大的
            total += len(labels)
            correct += (predicted == labels).sum()  # 预测对了的

        accuracy = 100 * correct / float(total)

    epoch_list.append(epoch)
    loss_list.append(loss)
    print('Epoch: {}  Loss: {}   Accuracy: {}%'.format(epoch, loss, accuracy))

plt.plot(epoch_list, loss_list)
plt.xlabel("Number of epoch")
plt.ylabel("Loss")
plt.title("Logistic Regression: Loss vs Number of epoch")
plt.savefig('baseline01lr.png')
plt.show()


# submit result
test_df = pd.read_csv('../drdata/test.csv', dtype=np.float32)
test_df = test_df.to_numpy() / 255.0
TestImages = torch.from_numpy(test_df)
test_loader = DataLoader(TestImages, batch_size=28000, shuffle=False)

with torch.no_grad():
    for ti in test_loader:
        tiv = Variable(ti.view(-1, 28*28))
        o = model(tiv)
        _, p = torch.max(o.data, 1)
        p = p.cpu()
        submission = pd.DataFrame({'ImageId': np.arange(1, (p.size(0) + 1)), 'Label': p})
        submission.to_csv("submission.csv", index=False)

