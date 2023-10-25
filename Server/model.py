import torch as pt

# 一个手写数字识别的model
class Model(pt.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = pt.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = pt.nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = pt.nn.Dropout2d()
        self.fc1 = pt.nn.Linear(320, 50)
        self.fc2 = pt.nn.Linear(50, 10)

    def forward(self, x):
        x = pt.nn.functional.relu(pt.nn.functional.max_pool2d(self.conv1(x), 2))
        x = pt.nn.functional.relu(pt.nn.functional.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = pt.nn.functional.relu(self.fc1(x))
        x = pt.nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return pt.nn.functional.log_softmax(x, dim=1)

    def train(self, train_loader, optimizer, epoch):
        self.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = self(data)
            loss = pt.nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

    def update_model_with_gradients(self, gradients, optimizer):
        # 将梯度应用到模型参数上
        for param, grad in zip(self.parameters(), gradients):
            param.data -= optimizer.lr * grad

        # 使用优化器的其他功能来更新参数
        optimizer.step()