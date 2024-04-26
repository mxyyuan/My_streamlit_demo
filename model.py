import torch
import torch.nn as nn
import numpy as np
# 网络定义
class CNNnet_cifar10(nn.Module):
    def __init__(self):
        super(CNNnet_cifar10,self).__init__()\
        #1*32*32
        self.conv1=torch.nn.Conv2d(3,32,3,1,1)
        self.relu1=torch.nn.ReLU()
        self.maxpool1=torch.nn.MaxPool2d(kernel_size = 2, stride=2)
   #input channels,output channels, kernel size,stride,padding,activation function,maxpooling
    #一个卷积模块
        #32*16*16
        self.conv2 = torch.nn.Conv2d(32,64,3,1,1)
        self.relu2 = torch.nn.ReLU()
        self.maxpool2= torch.nn.MaxPool2d(kernel_size = 2, stride=2)
       #64*8*8
        self.conv3 = torch.nn.Conv2d(64,64,3,1,1)
        self.relu3= torch.nn.ReLU()
        #torch.nn.MaxPool2d(kernel_size = 2, stride=2)

        self.mlp1 = torch.nn.Linear(64*8*8,128)
        self.mlp2 = torch.nn.Linear(128,10)#分为10类
        #self.output= torch.nn.Softmax()
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.mlp1(x.reshape(x.size(0),-1))#x.size(0) represents the batch size
        x = self.relu3(x)
        x = self.mlp2(x)
        #x = self.output(x)
        return x