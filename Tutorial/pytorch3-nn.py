import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets

train = datasets.MNIST("",train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("",train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))

#batch_size is how many at a time you can to pass into model
trainset= torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset= torch.utils.data.DataLoader(test,batch_size=10,shuffle=False)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #28*28 because images are 28x28
        #fully connected layer = fcl
        self.fc1=nn.Linear(28*28,64) #28*28 in, 64 out
        self.fc2=nn.Linear(64,64)#has to be 64 because fcl1 outputs 64
        self.fc3=nn.Linear(64,64)
        self.fc4=nn.Linear(64,10) #output 10 because 0-9

    def forward(self,x):
        #F.relu is rectified linear activation function
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=self.fc4(x) #dont run on output layer

        return F.log_softmax(x,dim=1) #apply log softmax instead

net=Net()
#print(net)

#random image
X = torch.rand((28,28))
X=X.view(1,28*28)#resize, -1 says that input is unknown shape

output=net(X)
print(output)
