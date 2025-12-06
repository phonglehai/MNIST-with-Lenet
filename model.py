
import torch.nn as nn 
class LenetClassifier(nn.Module):
  def __init__(self,num_classes):
    super().__init__()
    self.conv1 = nn.Conv2d(1,6,kernel_size = 5,padding='same')
    self.avgpol1 = nn.AvgPool2d(2,2) 
    self.conv2 = nn.Conv2d(6,16,5)
    self.avgpol2 = nn.AvgPool2d(2,2)
    self.flaten = nn.Flatten()
    self.relu = nn.ReLU()
    self.linear0 = nn.Linear(16*5*5,120)
    self.linear1 = nn.Linear(120,84)
    self.linear2 = nn.Linear(84,num_classes)
  def forward(self,x):
    x = self.conv1(x)

    x = self.avgpol1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.avgpol2(x)
    x = self.relu(x)
    x = self.flaten(x)
    x = self.linear0(x)
    x = self.linear1(x)
    x = self.linear2(x)
    return x 
