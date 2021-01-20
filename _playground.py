# %%
import numpy as np
import tunnel_control
import importlib
import time
from matplotlib import pyplot as plt
# %%

folder_path = 'data/2021-01-14'

t = time.time()
data_all = drag_analysis.analyze_experiment_set(folder_path,update=True,load_temperature=False)

elapsed = time.time() - t
print('Elapsed time: %.2f sec'%elapsed)
# %%
importlib.reload(drag_analysis)
drag_analysis.plot_data(data_all)

# plt.xscale('log')
# plt.yscale('log')


# %%
flat_keys = [x for x in data_all.keys() if x.startswith('Flat')]
drag_analysis.plot_selected(data_all,flat_keys)

# %%
importlib.reload(drag_analysis)
drag_analysis.plot_nu(data_all)
# %%
importlib.reload(drag_analysis)
drag_analysis.plot_Re_Drag(data_all)
# %%
importlib.reload(drag_analysis)

# %%
keys_1 = [x for x in data_all.keys() if x.startswith('1_1_1')]
keys_2 = [x for x in data_all.keys() if x.startswith('2_1.5_1.5')]
flat_keys = [x for x in data_all.keys() if x.startswith('Flat')]

# %%
data_111 = {k:data_all[k] for k in keys_1}
data_21515 = {k:data_all[k] for k in keys_2}
data_flat = {k:data_all[k] for k in flat_keys}
# %%
drag_analysis.plot_Re_Drag({k:data_all[k] for k in keys_2}) 
plt.xscale('log')
plt.yscale('log')
# %%
drag_analysis.plot_Re_Drag({k:data_all[k] for k in flat_keys}) 
plt.xscale('log')
plt.yscale('log')

# %%
importlib.reload(drag_analysis)
drag_analysis.plot_power(data_flat)
plt.grid()
# %%
drag_analysis.plot_power(data_21515)
plt.grid()
# %%
drag_analysis.plot_power(data_111)
plt.grid()

# %%
importlib.reload(drag_analysis)
t = drag_analysis.data_binning(data_111,overlap=False)
t = drag_analysis.data_binning(data_21515,overlap=False)
t = drag_analysis.data_binning(data_flat,overlap=False)

# plt.xscale('log')
# plt.yscale('log')
# %%
importlib.reload(drag_analysis)
drag_analysis.plot_with_std(data_111)
# %%
drag_analysis.plot_with_std(data_21515)
# %%
drag_analysis.plot_with_std(data_flat)

# %%
importlib.reload(drag_analysis)
folder_path = 'data/2021-01-12'
drag_analysis.inspect_all_exp(folder_path,update = False)
# %%
fig,ax = plt.subplots(3,1)
ax[0].plot([1,1],[0,2])

# %%
