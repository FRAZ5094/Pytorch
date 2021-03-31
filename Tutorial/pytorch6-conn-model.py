import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from time import perf_counter
from sigfig import round


if torch.cuda.is_available():
    device=torch.device("cuda:0")
    #print(f"running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device=torch.device("cpu")
    print("running on cpu")

#device=torch.device("cpu")

training_data=np.load("training_data.npy",allow_pickle=True)
#print(len(training_data))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,32,5)#input/output/kernal size (matrix size)
        self.conv2=nn.Conv2d(32,64,5)
        self.conv3=nn.Conv2d(64,128,5)

        x=torch.randn(50,50).view(-1,1,50,50)#-1 represents not knowing how many feature sets there are for the 1,50,50 tensor
        self._to_linear=None
        self.convs(x)#running fake data through to findout the output size

        self.fcl1=nn.Linear(self._to_linear,512)
        self.fcl2=nn.Linear(512,2)


    def convs(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2)) #(2,2) max maxing pooling
        x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv3(x)),(2,2))

        #print(x[0].shape)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        
        return x

    def forward(self,x):
        x=self.convs(x)
        x=x.view(-1,self._to_linear) #flatten
        x=F.relu(self.fcl1(x))
        x=self.fcl2(x)
        return F.softmax(x,dim=1)



net=Net().to(device)

optimizer=optim.Adam(net.parameters(),lr=0.001)
loss_function= nn.MSELoss()

X=torch.Tensor([i[0] for i in training_data]).view(-1,50,50)#make data tensor and resize
X=X/255.0 #normalize from 0-255 to 0-1
y=torch.Tensor([i[1] for i in training_data])

n=len(X)-1000

train_X=X[:n]
train_y=y[:n]

#make sure train and test data is different

test_X=X[n:]
test_y=y[n:]

BATCH_SIZE=100
EPOCHS=5
losses=[]
start=perf_counter()
for epoch in range(EPOCHS):
    print("epoch: ",epoch)
    for i in tqdm(range(0,len(train_X),BATCH_SIZE)): #from zero to length of data in BATCH_SIZE chunks
        #print(i,i+BATCH_SIZE)
        batch_X=train_X[i:i+BATCH_SIZE].view(-1,1,50,50).to(device)
        batch_y=train_y[i:i+BATCH_SIZE].to(device)

        net.zero_grad()
        output=net(batch_X)
        loss=loss_function(output,batch_y)
        loss.backward()
        optimizer.step()


end=perf_counter()
print(f"trained for {EPOCHS} epochs in {round(end-start,2)} seconds")

correct=0
total=0
print("calculating accuracy...")
with torch.no_grad():
    for i in tqdm(range(len(test_X))):
        real_class=torch.argmax(test_y[i]).to(device)
        net_out=net(test_X[i].view(-1,1,50,50).to(device))[0]
        predicted_class=torch.argmax(net_out)
        if predicted_class==real_class:
            correct+=1
        total+=1
print("Accuracy:", round(correct/total,3))
