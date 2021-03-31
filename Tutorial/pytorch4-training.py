import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
import torch.optim as optim
from time import perf_counter
from sigfig import round
import matplotlib.pyplot as plt

train = datasets.MNIST("",train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("",train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))

#batch_size is how many at a time you can to pass into model
trainset= torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset= torch.utils.data.DataLoader(test,batch_size=10,shuffle=True)

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


optimizer=optim.Adam(net.parameters(),lr=0.001) #lr is learning rate

EPOCHS=10

for epoch in range(EPOCHS):
    start=perf_counter()
    for data in trainset:
        X,y=data
        net.zero_grad() #everytime you pass data in 
        output=net(X.view(-1,28*28))
        loss=F.nll_loss(output,y) #use if output is single scalar value
        loss.backward() #back propigation
        optimizer.step() #adusts weights

    end=perf_counter()
    print(loss)
    print(f"took {round(end-start,2)}")


correct=0
total=0

with torch.no_grad():#don't want to change gadients, just test model
    for data in testset:
        X,y = data
        output = net(X.view(-1,28*28))
        for idx,i in enumerate(output):
            if torch.argmax(i)==y[idx]:
                correct+=1
            total+=1

print("Accuracy: ",round(correct/total,2))
"""
n=4
plt.imshow(X[n].view(28,28),cmap='gray',vmin=0,vmax=1)
plt.show()

print(torch.argmax(net(X[n].view(-1,784))[0]))

"""