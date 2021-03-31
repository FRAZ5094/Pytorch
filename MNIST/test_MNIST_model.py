import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from MNIST_model import Net
import matplotlib.pyplot as plt
import cv2
import glob as glob
from sigfig import round
from time import perf_counter

device=torch.device("cuda:0")

loaded_checkpoint=torch.load("MNIST.pth")

epoch=loaded_checkpoint["epoch"]

net= Net()
optimizer=optim.Adam(net.parameters(),lr=0.001)

net.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optim_state"])
net.to(device)

files=glob.glob("Testing/*")

for path in files:

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img,(28,28))
    img = cv2.bitwise_not(img)
    plt.imshow(img,cmap="gray")
    plt.show()

    X=torch.Tensor(img).view(-1,28,28)
    X/=255.0
    X=X.view(-1,1,28,28).to(device)
    start=perf_counter()
    net_out=net(X)[0]
    end=perf_counter()
    print(f"Took: {round(end-start,3)}")
    predicted_class=torch.argmax(net_out)

    print(f"{predicted_class}")
    print(f"{net_out[predicted_class]*100}%")

