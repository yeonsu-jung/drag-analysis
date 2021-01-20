# %%
import numpy as np
import tunnel_control
import importlib
import time
from matplotlib import pyplot as plt
# %%

folder_path = 'data/2021-01-14'

t = time.time()
data_all = tunnel_control.analyze_experiment_set(folder_path,update=True,load_temperature=False)

elapsed = time.time() - t
print('Elapsed time: %.2f sec'%elapsed)
# %%
importlib.reload(tunnel_control)
tunnel_control.plot_data(data_all)

# plt.xscale('log')
# plt.yscale('log')


# %%
flat_keys = [x for x in data_all.keys() if x.startswith('Flat')]
tunnel_control.plot_selected(data_all,flat_keys)

# %%
importlib.reload(tunnel_control)
tunnel_control.plot_nu(data_all)
# %%
importlib.reload(tunnel_control)
tunnel_control.plot_Re_Drag(data_all)
# %%
importlib.reload(tunnel_control)

# %%
keys_1 = [x for x in data_all.keys() if x.startswith('1_1_1')]
keys_2 = [x for x in data_all.keys() if x.startswith('2_1.5_1.5')]
flat_keys = [x for x in data_all.keys() if x.startswith('Flat')]

# %%
data_111 = {k:data_all[k] for k in keys_1}
data_21515 = {k:data_all[k] for k in keys_2}
data_flat = {k:data_all[k] for k in flat_keys}
# %%


tunnel_control.plot_Re_Drag(})
plt.xscale('log')
plt.yscale('log')
# %%
tunnel_control.plot_Re_Drag({k:data_all[k] for k in keys_2}) 
plt.xscale('log')
plt.yscale('log')
# %%
tunnel_control.plot_Re_Drag({k:data_all[k] for k in flat_keys}) 
plt.xscale('log')
plt.yscale('log')

# %%
importlib.reload(tunnel_control)
tunnel_control.plot_power(data_flat)
plt.grid()
# %%
tunnel_control.plot_power(data_21515)
plt.grid()
# %%
tunnel_control.plot_power(data_111)
plt.grid()

# %%
importlib.reload(tunnel_control)
t = tunnel_control.data_binning(data_111,overlap=False)
t = tunnel_control.data_binning(data_21515,overlap=False)
t = tunnel_control.data_binning(data_flat,overlap=False)

# plt.xscale('log')
# plt.yscale('log')
# %%
importlib.reload(tunnel_control)
tunnel_control.plot_with_std(data_111)
# %%
tunnel_control.plot_with_std(data_21515)
# %%
tunnel_control.plot_with_std(data_flat)

# %%
importlib.reload(tunnel_control)
folder_path = 'data/2021-01-12'
tunnel_control.inspect_all_exp(folder_path,update = False)
# %%
fig,ax = plt.subplots(3,1)
ax[0].plot([1,1],[0,2])

# %%
