def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

class Conv(nn.Module):
    def __init__(self):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.fc1 = nn.Linear(4*4*50, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(datloader,lossfunc,opti)
    n_epochs = 2
    for epoch in range(n_epochs):

        totloss=0.0
        for i,data in enumerate(datloader,0):
            inputs,labels = data

            opti.zero_grad()

            outputs = conv(inputs)

            loss = lossfunc(outputs, labels)
            loss.backward()
            opti.step()

            totloss += loss.item()

        print('loss = '+str(totloss))

    print('Finished Training')

def test(datloader)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in datloader:
            images, labels = data
            outputs = conv(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100.0 * correct / total))
