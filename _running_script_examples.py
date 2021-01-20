# %%
import drag_analysis
import os

drag_data_folder_path = 'D:/Rowland/data/drag-data' # replace with your drag data folder path
date_folder_path = '2021-01-14'
entire_folder_path = os.path.join(dropbox_folder_path,drag_data_folder_path,date_folder_path)

# %%
# analyzing entire folder and store the result in 'data_all' dictionary
# this output dictionary contains experiment condition string as a key, and numpy array [U, drag, standard error, standard deviation, nu(kinematic viscosity)] as a value.

data_all = drag_data_analysis.analyze_experiment_set(folder_path,update=False,load_temperature=True)

# %%
# plot all result data in 'data_all'
drag_analysis.plot_data(data_all)

# %%
# plot kinematic viscosity in 'data_all'
drag_analysis.plot_nu(data_all)

# %%
# plot Drag vs Re plot in 'data_all'
drag_analysis.plot_Re_Drag(data_all)

# select part of data using list comprehension
keys_1_1_1_20 = [x for x in data_all.keys() if x.startswith('1_1_1')]
keys_2_15_15_20 = [x for x in data_all.keys() if x.startswith('2_1.5_1.5')]
keys_flat_10 = [x for x in data_all.keys() if x.startswith('Flat_10')]
keys_flat_20 = [x for x in data_all.keys() if x.startswith('Flat_20')]

data_1_1_1_20 = {k:data_all[k] for k in keys_1_1_1_20}
data_2_15_15 = {k:data_all[k] for k in keys_2_15_15_20}
data_flat_10 = {k:data_all[k] for k in keys_flat_10}
data_flat_20 = {k:data_all[k] for k in keys_flat_20}

# %%
# binding data for repeated conditions (under development)

t = drag_analysis.data_binning(data_111,overlap=False)
t = drag_analysis.data_binning(data_21515,overlap=False)
t = drag_analysis.data_binning(data_flat,overlap=False)
# %%
# plot with standard deviation, not with standard error of means
drag_analysis.plot_with_std(data_111)
drag_analysis.plot_with_std(data_21515)
drag_analysis.plot_with_std(data_flat)

# %%
# inspecting data by plotting all the raw data
# this takes some time

entire_folder_path = 'data/2021-01-12'
drag_analysis.inspect_all_exp(entire_folder_path,update = False)
