import csv
import re
import json
import re
import numpy as np
from matplotlib import pyplot as plt
import math

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

def f(x):
   return 8*x
def g(x):
   return 12*(2**(1/2)+1)/2**(1/2)*x**(1/2)

def h(x):
   return 0.01*(200*math.log10(200+x))**(1/2)

def H(x,T):
    y = 0
    for _ in range(T):
       new_y = h(x)
       y += new_y
    return y
    
    
x = np.linspace(1, 50, 1000)
plt.plot(x, f(x), color='navy', label="linear regression")
plt.plot(x, g(x), color='tomato', label="LSTM")

H_list = []

for i in range(len(x)):
    H_list.append(H(x[i],100))
plt.plot(x, H_list, color='limegreen', label="Decision Tree ensemble model")

   
# plt.plot(x, H(x,1000), color='green')

plt.xlabel("Number of features d")
plt.ylabel("Error")
plt.legend()
plt.savefig("theory.png",dpi=500)
# plt.show()