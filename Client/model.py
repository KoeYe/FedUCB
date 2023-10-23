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
    
    # 定义train方法，这里需要通过