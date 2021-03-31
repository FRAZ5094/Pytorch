import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
if torch.cuda.is_available():
    device=torch.device("cuda:0")
    print(f"running on GPU: {torch.cuda.get_device_name(0)}")
else:
    device=torch.device("cpu")
    print("running on cpu")



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,32,5)
        self.conv2=nn.Conv2d(32,64,5)
        self.conv3=nn.Conv2d(64,128,3)

        x=torch.randn(28,28).view(-1,1,28,28)
        self._to_linear=None
        self.convs(x)

        self.fcl1=nn.Linear(self._to_linear,512)
        self.fcl2=nn.Linear(512,10)


    def convs(self,x):
        x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv2(x)),(2,2))
        x=F.max_pool2d(F.relu(self.conv3(x)),(2,2))


        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        
        return x

    def forward(self,x):
        x=self.convs(x)
        x=x.view(-1,self._to_linear)
        x=F.relu(self.fcl1(x))
        x=self.fcl2(x)
        return F.softmax(x,dim=1)

def fwd_pass(x,y,train=False):
    if train:
        net.zero_grad()
    outputs=net(x)
    matches=[torch.argmax(i)==torch.argmax(j) for i,j in zip(outputs,y)]
    acc=matches.count(True)/len(matches)
    loss=loss_function(outputs,y)
    if train:
        loss.backward()
        optimizer.step()
    return acc,loss

def test(size=32):
    random_start=np.random.randint(len(test_x)-size)
    x,y = test_x[random_start:random_start+size],test_y[random_start:random_start+size]
    with torch.no_grad():
        val_acc,val_loss=fwd_pass(x.view(-1,1,28,28).to(device),y.to(device))
    return val_acc,val_loss

if __name__=="__main__":

    net=Net().to(device)

    optimizer=optim.Adam(net.parameters(),lr=0.001)
    loss_function= nn.MSELoss()

    training_data=np.load("training_data.npy",allow_pickle=True)
    x=[]
    y=[]
    for data in training_data:
        x.append(data[0])
        y.append(data[1])

    x=torch.Tensor(x)
    y=torch.Tensor(y)

    testing_data=np.load("testing_data.npy",allow_pickle=True)
    test_x=[]
    test_y=[]
    for data in testing_data:
        test_x.append(data[0])
        test_y.append(data[1])

    test_x=torch.Tensor(test_x)
    test_y=torch.Tensor(test_y)

    BATCH_SIZE=100
    EPOCHS=15

    MODEL_NAME=f"MNIST-{int(time.time())}"
    print(MODEL_NAME)

    with open("model.log","a") as f:
        for epoch in range(EPOCHS):
            print("epoch: ",epoch)
            for i in tqdm(range(0,len(x),BATCH_SIZE)):
                batch_x=x[i:i+BATCH_SIZE].view(-1,1,28,28).to(device)
                batch_y=y[i:i+BATCH_SIZE].to(device)

                acc,loss=fwd_pass(batch_x,batch_y,train=True)
                if i%50==0:
                    val_acc,val_loss=test(size=32)
                    f.write(f"{MODEL_NAME},{round(time.time(),3)},{round(float(acc),2)},{round(float(loss),4)},{round(float(val_acc),2)},{round(float(val_loss),4)}\n")

    checkpoint={
        "epoch":epoch+1,
        "model_state": net.state_dict(),
        "optim_state": optimizer.state_dict(),
    }

    torch.save(checkpoint,"MNIST.pth")