import matplotlib.pyplot as plt
import pandas as pd
%matplotlib notebook
data=pd.read_csv("model.log",header=None)

plt.plot(data[2])
plt.plot(data[4])
plt.ylim(0.9,1)
plt.show()