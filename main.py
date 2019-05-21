import torch
import numpy as np
import torch.nn as nn
import torchvision as tv
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from utils.py import *

tf = tv.transforms.Compose([tv.transforms.ToTensor(),tv.transforms.Normalize(mean=(0.5,),std=(0.5,))])
trainset = tv.datasets.MNIST(root='./data',train=True,download=True,transform=tf)
trainloader = torch.utils.data.DataLoader(trainset,shuffle=True,batch_size=4,num_workers=2)
testset = tv.datasets.MNIST(root='./data',train=False,download=True,transform=tf)
testloader = torch.utils.data.DataLoader(testset,shuffle=True,batch_size=4,num_workers=2)

classes = ('0','1','2','3','4','5','6','7','8','9')

myiter = iter(trainloader)
image,label = myiter.next()
imshow(tv.utils.make_grid(image))
print(''.join('%2s' % classes[l] for l in label))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(conv.parameters(), lr=0.01)

train(trainloader,criterion,optimizer)

test(testloader)

myiter2 = iter(testloader)
image,label = myiter2.next()
imshow(tv.utils.make_grid(image))
outputs = conv(image)
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' ',[classes[p] for p in predicted])
