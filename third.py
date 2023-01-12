# import cv2
# import torch
# from torch import torch_version
# from torch import nn
# from torch.nn import functional as F
#
# input_image = torch.tensor([[1, 2, 0, 3, 1],
#                             [0, 1, 2, 3, 1],
#                             [1, 2, 1, 0, 0],
#                             [5, 2, 3, 1, 1],
#                             [2, 1, 0, 1, 1]])
# input_image = torch.reshape(input_image, (1, 1, 5, 5,))
# print(input_image.shape)
#
# print()
# kernel = torch.tensor([[1, 2, 1],
#                        [0, 1, 0],
#                        [2, 1, 0]])
# kernel = torch.reshape(kernel, (1, 1, 3, 3,))
# print(kernel.shape)
# output = F.conv2d(input_image, kernel)
#
# print(output)
from torchvision import transforms
from torchvision import datasets
dataset = datasets.CIFAR10("datas",train=False,transform=transforms.ToTensor(),download=True)

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
