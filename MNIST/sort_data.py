import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from tqdm import tqdm
import torch
from sigfig import round


data=pd.read_csv("mnist_test.csv")
training_data=[]
count={0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0,9:0}

for i in tqdm(range(len(data))):
    row=data.iloc[i]
    img=np.array(row[1::])
    label=int(row[0])
    img=img/255.0
    img=img.reshape(28,28)
    training_data.append([np.array(img),np.eye(10)[label]])
    count[label]+=1


sum=0
for key in count:
    sum+=count[key]

for key in count:
    print(f"{key}: {round(count[key]/sum*100,2)}%")

np.random.shuffle(training_data)
np.save("testing_data.npy",training_data)
print("data saved")
print(len(training_data))





