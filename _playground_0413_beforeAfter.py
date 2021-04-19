
import numpy as np

before = np.loadtxt('C:/Users/Water Tunnel/Documents/GitHub/tunnel_control/_data/drag-data/2021-04-13/Flat_10 (black)/_zero.csv')
after = np.loadtxt('C:/Users/Water Tunnel/Documents/GitHub/tunnel_control/_data/drag-data/2021-04-13/Flat_10 (black) (after experiment)/_zero.csv')
# %%
from matplotlib import pyplot as plt
plt.plot(before)
plt.plot(after)

# %%
plt.plot([0,300000],[np.mean(before),np.mean(before)],'r')
plt.plot([0,300000],[np.mean(after),np.mean(after)],'b')
# %%
(np.mean(after) - np.mean(before))/0.013156