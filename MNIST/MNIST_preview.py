import numpy as np
import matplotlib.pyplot as plt 
import torch

testing_data=np.load("testing_data.npy",allow_pickle=True)
test_x=[]
test_y=[]
for data in testing_data:
    test_x.append(data[0])
    test_y.append(data[1])

number=8

count=0
for i in range(len(test_x)):
    if test_y[i][number]==1.:
        plt.imshow(test_x[i],cmap="gray")
        plt.show()
        count+=1
    if count==10 or i>1000:
        break


    