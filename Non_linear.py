import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./datas_cifar10", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64)

writer = SummaryWriter("./logs_cifar10")


class TuTu(nn.Module):
    def __init__(self):
        super(TuTu, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.sigmoid(input)
        return output


tutu = TuTu()

step=0

for data in dataloader:
    img, target = data
    writer.add_images("imags",img,global_step=step)
    output=tutu(img)
    writer.add_images("output",output,global_step=step)
    step+=1
writer.close()