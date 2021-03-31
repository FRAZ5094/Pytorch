import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from save_model import Net
import matplotlib.pyplot as plt
import cv2
import glob as glob

device=torch.device("cuda:0")

loaded_checkpoint=torch.load("DogsVSCats.pth")

epoch=loaded_checkpoint["epoch"]

net= Net()
optimizer=optim.Adam(net.parameters(),lr=0.001)

net.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optim_state"])
net.to(device)

files=glob.glob("PetImages/Testing/*")

for path in files:

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(50,50))
    plt.imshow(img,cmap="gray")
    plt.show()

    X=torch.Tensor(img).view(-1,50,50)
    X/=255.0

    net_out=net(X.view(-1,1,50,50).to(device))[0]
    predicted_class=torch.argmax(net_out)

    if predicted_class==0:
        print("cat")
        print(f"{net_out[0]*100}%")
    else:
        print("dog")
        print(f"{net_out[1]*100}%")



