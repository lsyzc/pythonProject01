import torch
from torch.utils.data import DataLoader
import torchvision
from torch import nn

train_dataset = torchvision.datasets.CIFAR10("./CIFA10_train", train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10("./CIFA10_test", train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=64)

test_dataloader = DataLoader(test_dataset, batch_size=64)

print(type(train_dataset[0][0]))
print(train_dataset[0][0].shape)


# 定义模型结构
class TuTu(nn.Module):
    def __init__(self):
        super(TuTu, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5), padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


# 实例化
tutu = TuTu()

if torch.cuda.is_available():
    tutu = tutu.cuda()  # 1.模型cuda()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():  # 2.损失函数cuda()
    loss_fn.cuda()

# 优化器
learning_rate = 1e-2

optimizer = torch.optim.SGD(tutu.parameters(), lr=learning_rate)


epoch = 10

for i in range(epoch):  # 3. 数据cuda()
    train_step = 0
    print("第{}轮训练".format(i))
    for image, target in train_dataloader:
        if torch.cuda.is_available():
            image = image.cuda()
            target = target.cuda()
        output = tutu(image)
        loss = loss_fn(output, target)
        # 优化器使用
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_step += 1
        if train_step % 100 == 0:
            print("第{}次训练完成\n".format(train_step))
