from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer = SummaryWriter('logs')
for i in range(100):
    writer.add_scalar('y=2x图像',2*i,i)
writer.close()