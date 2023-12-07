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
            param.grad = grad

        # 使用优化器的其他功能来更新参数
        optimizer.step()

    def test(self, test_data):
        """
            计算平均正确率和loss
            test_data['y']是一个list，里面是每个图片对应的数字
            test_data['x']是一个list，里面是每个图片的像素值
            test_data['x'][0]是一个28*28的图片拉长之后的一维list，长度是784，注意变换纬度
        """
        # 计算平均正确率
        correct = 0
        # 计算loss
        loss = 0
        # 遍历每个图片
        for (y, x) in zip(test_data['y'], test_data['x']):
            # 将data从（784，）变成（28，28）
            data = pt.tensor(x).view(1, 28, 28)
            # 计算output
            output = self(data)
            loss += pt.nn.CrossEntropyLoss()(output, pt.tensor([y], dtype=pt.long))
            # 计算正确率
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(pt.tensor([y]).data.view_as(pred)).sum()

        # 返回正确率和loss
        return correct / len(test_data['y']), loss / len(test_data['y'])