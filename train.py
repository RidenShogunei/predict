import torch
import torch.nn as nn
import torch.optim as optim
from data import dataget
from modle import CNN

# 初始化模型
model = CNN()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
# 获得数据
x_train, y_train = dataget()
print(x_train.shape)
# 将输入数据转换为Tensor格式
x_train_tensor = torch.Tensor(x_train)
x_train_tensor = torch.unsqueeze(x_train_tensor, 1)
x_train_tensor = x_train_tensor.transpose(1, 2)
x_train_tensor = x_train_tensor.reshape([x_train_tensor.shape[0] // 5, 36*5])
print(x_train_tensor.shape)  # 打印转置前的形状
y_train_tensor = torch.Tensor(y_train)
y_train_tensor = torch.unsqueeze(y_train_tensor, 1)
y_train_tensor = y_train_tensor.reshape([y_train_tensor.shape[0] // 5, 1*5])
print(y_train_tensor.shape)  # 打印转置前的形状

# 进行模型训练
for epoch in range(num_epochs):
    # 前向传播
    outputs = model(x_train_tensor)
    outputs = outputs.float()
    print(outputs)
    # 重新计算目标张量
    y_train_tensor = torch.Tensor(y_train)
    y_train_tensor = torch.unsqueeze(y_train_tensor, 1)

    # 调整形状
    y_train_tensor = y_train_tensor.float()
    y_train_tensor = y_train_tensor.view(-1, 1)

    loss = criterion(outputs, y_train_tensor)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 打印训练信息
    if (epoch + 1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
