import sys,pygame
from pygame.locals import *
import numpy as np
import cv2
import torch
import torch.optim as optim
from MNIST_model import Net
import matplotlib.pyplot as plt
pygame.init()

width=140
height=width

screen = pygame.display.set_mode((width,height))

screen.fill((0,0,0))

brush = pygame.image.load("brush.png")
brush=pygame.transform.scale(brush,(10,10))

device=torch.device("cuda:0")

loaded_checkpoint=torch.load("MNIST.pth")

epoch=loaded_checkpoint["epoch"]

net= Net()
optimizer=optim.Adam(net.parameters(),lr=0.001)

net.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optim_state"])
net.eval()
net.to(device)

pygame.display.update()

clock= pygame.time.Clock()
yes=True
z = 0
while 1:
    clock.tick(240)
    x,y=pygame.mouse.get_pos()
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            sys.exit()
        if event.type==MOUSEBUTTONDOWN:
            z=1
        if event.type==MOUSEBUTTONUP:
            z=0
        if event.type==KEYDOWN:
            if event.key==K_w:
                print("guessing...")

                pixels=[]
                for i in range(width):
                    y=[]
                    for j in range(height):
                        pixel=screen.get_at((j,i))[:3]
                        if pixel[0]==255:
                            y.append(1)
                        else:
                            y.append(0)
                    pixels.append(y)

                array = np.array(pixels, dtype=np.uint8)
                img = cv2.resize(array,(28,28))
                #plt.imshow(img,cmap="gray")
                #plt.show()
                X=torch.Tensor(img).view(-1,28,28)
                X=X.view(-1,1,28,28).to(device)

                with torch.no_grad():
                    net_out=net(X)[0]
                predicted_class=torch.argmax(net_out)
                print(f"{predicted_class}")
                print(f"{net_out[predicted_class]*100}%")

            if event.key==K_c:
                print("clearning")
                screen.fill((0,0,0))
                pygame.display.update()
            
        if z==1:
            screen.blit(brush,(x-10,y-10))
            pygame.display.update()

    

    
