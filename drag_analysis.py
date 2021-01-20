# %%

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import time
import os
import sys

from scipy.optimize import curve_fit
from pathlib import Path
from PIL import Image
from scipy.optimize import curve_fit

# %%
def check_average_value(path_in):
    df = pd.read_csv(path_in,comment='#',header=None).to_numpy()
    return (np.mean(df))

def temp_to_kinematic_viscosity(temp):
    rho = (999.83952 + 16.945176*temp - 7.9870401e-3*temp**2 - 46.170461e-6*temp**3 + 105.56302e-9*temp**4 - 280.54253e-12*temp**5)/(1 + 16.879850e-3*temp)
    
    if temp > 20:
        A = 1301/(998.333 + 8.1855*(temp-20) + 0.00585*(temp-20)**2) - 1.30223
        mu = 1e-3*10**(A)
    else:
        B = (1.3272*(20-temp) - 0.001053*(temp-20)**2)/(temp+105)
        mu = (1.002e-3)*10**(B)
    
    # from (R. C. Weast, 1983, CRC Handbook of Chemistry and Physics, 64th edition, CRC Press, Boca Raton, FL

    return mu/rho

def minute_rounder(t):
        # Rounds to nearest minute by adding a timedelta minute if sec >= 30
        return (t.replace(second=0, microsecond=0, minute=t.minute, hour=t.hour)
               +datetime.timedelta(minutes=t.second//30))

def find_temperature(start_datetime,end_datetime):
    # Note that there's a ~15 sec time difference between PC and thermostat times (2021-01-11 - YJ)
    df = pd.read_csv('temperature/latest.csv',skiprows=14,usecols=range(1,4))

    # print(type(start_datetime))
    print(start_datetime, end_datetime)

    start_datetime_object = minute_rounder(datetime.datetime.strptime(start_datetime,"%Y-%m-%d_%H:%M:%S"))
    end_datetime_object = minute_rounder(datetime.datetime.strptime(end_datetime,'%Y-%m-%d_%H:%M:%S'))

    # start_datetime_object = minute_rounder(datetime.datetime.strptime(start_datetime,'%Y-%m-%d_%H:%M:%S'))
    # end_datetime_object = minute_rounder(datetime.datetime.strptime(end_datetime,'%Y-%m-%d_%H:%M:%S'))

    start_date = start_datetime_object.strftime("%m/%d/%y")
    end_date = end_datetime_object.strftime("%m/%d/%y")

    start_time = start_datetime_object.strftime("%H:%M")
    end_time = end_datetime_object.strftime("%H:%M")

    try:
        start_temperature = df.loc[(df['Date'] == start_date) & (df['Time'] == start_time)].iloc[0,0]
        end_temperature = df.loc[(df['Date'] == end_date) & (df['Time'] == end_time)].iloc[0,0]
    except: # which error?
        print('No temperature data.')
        sys.exit(1)

    return start_temperature, end_temperature

    # if time span exceed 1 min, this code should be modified to get temperature array, having size larger than 2. - YJ

def analyze_new_folder(exp_path_in,load_temperature=False):
    zero_file_path = os.path.join(exp_path_in,'_zero.csv')
    try:
        zero = np.mean(pd.read_csv(zero_file_path,comment='#',header=None).to_numpy())
    except:
        print('No zeroing result file. Do zeroing first.')
        sys.exit(1)

    csv_list = [x for x in os.listdir(exp_path_in) if not x.startswith('_')]
    data_array = []
    for csv_file in csv_list:
        csv_file_path = os.path.join(exp_path_in, csv_file)
        df = pd.read_csv(csv_file_path,comment='#',header=None).to_numpy() - zero        

        with open(csv_file_path,'r') as f:
            start_time = f.readline()
            end_time = f.readline()

        if load_temperature:
            dummy, start_time = start_time.split('# ',1)
            dummy, end_time = end_time.split('# ',1)

            start_time = start_time.strip('\n')
            end_time = end_time.strip('\n')        

            start_temperature, end_temperature = find_temperature(start_time,end_time)        
            print(start_temperature,end_temperature)
            start_nu = temp_to_kinematic_viscosity(start_temperature)
            end_nu = temp_to_kinematic_viscosity(end_temperature)
        else:
            start_nu = 1e-6
            end_nu = 1e-6

        a,b,c,d = averaging(df,start_nu,end_nu)

        lhs,rhs = csv_file.split('.csv',1)
        U = 0.0485*float(lhs) - 0.0088
        data_array.append([U,a,b,c,d])
    data_array = np.array(data_array)
    data_array = data_array[np.argsort(data_array[:,0])]

    np.savetxt(os.path.join(exp_path_in, '_result.csv'),data_array,delimiter=',')
    return data_array

def analyze_a_folder(exp_path_in,update,load_temperature):
    if update == True:
        data_array = analyze_new_folder(exp_path_in,load_temperature)    
    elif check_result_file(exp_path_in):
        data_array = pd.read_csv(os.path.join(exp_path_in, '_result.csv'),header=None,comment='#').to_numpy()
    elif ~check_result_file(exp_path_in):
        data_array = analyze_new_folder(exp_path_in,load_temperature)    
    return data_array

def analyze_experiment_set(path_in,update=False,load_temperature=False):
    exp_list = [x for x in os.listdir(path_in) if not x.startswith('_')]
    data_all = {}
    for exp in exp_list:
        data_all[exp] = analyze_a_folder(os.path.join(path_in, exp),update=update,load_temperature=load_temperature)
    return data_all

def averaging(data_in,start_nu,end_nu,N=10):    
    data_reshaped = np.reshape(data_in,(N,int(data_in.size/N)))    
    
    means = np.mean(data_reshaped,axis=1)            
    all_std = np.std(data_in)
    
    mean_of_means = np.mean(means)
    error_of_means = np.std(means)
    
    # nu = 0.5*(start_nu + end_nu)
    nu = end_nu
    
    return mean_of_means, error_of_means, all_std, nu

def do_calibration(path_in):
    # temporary workaround for drag reading - Force unit conversion

    a = 0.013156
    return a

def check_result_file(folder_path_in):        
    result_file_path = os.path.join(folder_path_in,'_result.csv')
    if os.path.isfile(result_file_path):
        file_list = [x for x in os.listdir(folder_path_in) if not x.startswith('_')]
        saved_result = pd.read_csv(result_file_path,names=['']).to_numpy()
        if saved_result.size == len(file_list):
            return True
        else:
            return False
    else:
        return False

def inspect_new_exp(exp_path_in):
    csv_list = [x for x in os.listdir(exp_path_in) if not x.startswith('_')]    
    dummy_array = []
    for csv_file in csv_list:
        lhs,rhs = csv_file.split('.csv',1)
        dummy_array.append(float(lhs))
    sorted_arg = np.uint32(np.argsort(np.array(dummy_array)))
    csv_list = [csv_list[i] for i in sorted_arg]
    # print(csv_list)

    N = len(csv_list)
    i = 0
    fig, ax = plt.subplots(N+1,1,figsize=(12,N*3))
    for csv_file in csv_list:        
        df = pd.read_csv(os.path.join(exp_path_in,csv_file),comment='#',header=None).to_numpy()        
        lhs,rhs = csv_file.split('.csv',1)
        U = 0.0485*float(lhs) - 0.0088             
        ax[i+1].plot(df,label='%.2f m/s'%U)        
        ax[i+1].legend()
        # ax[i].set_ylim([np.mean(df)-0.5,np.mean(df)+0.5])
        ax[0].plot(df,label = '%.2f m/s'%U,zorder = N-i)
        i = i + 1                       
    
    plt.savefig(os.path.join(exp_path_in,'_inspection.png'))

def inspect_single_exp(exp_path_in):
    if os.path.isfile(os.path.join((exp_path_in,'_inspection.png'))):
        im = Image.open(os.path.join(exp_path_in,'_inspection.png'))    
        im.show()
    else:
        inspect_new_exp(exp_path_in)

def inspect_all_exp(folder_path_in,update = False):
    if update:            
        exp_list = [x for x in os.listdir(folder_path_in) if not x.startswith('_')]
    else:
        exp_list = [x for x in os.listdir(folder_path_in) if not x.startswith('_') and not os.path.isfile(os.path.join(folder_path_in,x,'_inspection.png'))]
    
    for exp in exp_list:
        if os.path.isfile( os.path.join(folder_path_in,exp) + '/_inspecttion.png' ):
            pass
        else:
            inspect_new_exp(os.path.join(folder_path_in,exp))

def plot_data(data_all):
    for k in data_all:
        U_array = data_all[k][:,0]
        drag_array = -data_all[k][:,1]
        error_array = (data_all[k][:,2] - data_all[k][0,1] + data_all[k][0,1])
        plt.errorbar(U_array,drag_array,error_array,fmt='o',label=k,capsize=5)
        plt.legend()

def plot_selected(data_all,keys):
    plot_data({k:data_all[k] for k in keys})    

def plot_data_2(data_all):
    for k in data_all:
        U_array = data_all[k][:,0]
        drag_array = -data_all[k][:,1]
        plt.plot(U_array,drag_array,'o-',label=k)
        plt.legend()

def plot_selected_2(data_all,keys):
    plot_data_2({k:data_all[k] for k in keys})

def convert_data(key,data_array):
    # length from k
    try:
        param_string_without_numbering, dummy = key.split(' ',1)
    except:
        param_string_without_numbering = key

    sample_length = float(param_string_without_numbering.split('_')[-1])*0.1 # in cm units
    
    U_array = data_array[:,0]
    Re_array = U_array * sample_length / data_array[:,4]
    drag_array = -data_array[:,1]/0.013156   
    e2_array = data_array[:,3]/0.013156

    return U_array, Re_array, drag_array, e2_array

def plot_Re_Drag(data_all):
    for k in data_all:
        U_array, Re_array, drag_array, e2_array = convert_data(k,data_all[k])
        plt.plot(Re_array,drag_array,'o',label=k)        
        plt.xlabel('Re')
        plt.ylabel('Drag (gf)')
        # plt.errorbar(U_array,drag_array,error_array,fmt='o',label=k,capsize=5)
        plt.legend()        

def plot_power(data_all):
    for k in data_all:
        U_array, Re_array, drag_array, e2_array = convert_data(k,data_all[k])
        log_x = np.log(Re_array)
        log_y = np.log(drag_array)

        n = (log_y[1:-1] - log_y[0:-2])/(log_x[1:-1] - log_x[0:-2])
        plt.plot(Re_array[0:-2],n,'o-',label=k)        
        
        plt.xlabel('Re')
        plt.ylabel('n')
        # plt.errorbar(U_array,drag_array,error_array,fmt='o',label=k,capsize=5)
    
    # plt.plot([0.5e6,2.25e6],[2,2],'k-.',label='2')
    plt.plot([0.5e6,2.25e6],[1.5,1.5],'k--',label='1.5')
    plt.legend()            

def plot_nu(data_all):
    for k in data_all:
        U_array = data_all[k][:,0]        
        nu_array = data_all[k][:,4]
        plt.plot(U_array,nu_array,'o-',label=k)
        plt.xlabel('U (m/s)')
        plt.ylabel('nu (m^2/s)')
        plt.legend()

def data_binning(grouped_data_dictionary,overlap=False):    
    first_key = list(grouped_data_dictionary.keys())[0]
    ncol = len(grouped_data_dictionary[first_key])
    nrow = len(grouped_data_dictionary.keys())    
    
    all_Re_array = np.zeros((nrow,ncol))
    all_drag_array = np.zeros((nrow,ncol))

    i = 0
    for k in grouped_data_dictionary:
        U_array, Re_array, drag_array, e2_array = convert_data(k,grouped_data_dictionary[k])
        all_Re_array[i,:] = Re_array
        all_drag_array[i,:] = drag_array
        if overlap:
            plt.plot(Re_array,drag_array,'o',label=k)
        i = i + 1

    averaged_Re_array = np.mean(all_Re_array,axis=0)
    averaged_drag_array = np.mean(all_drag_array,axis=0)

    error_Re_array = np.std(all_Re_array,axis=0)
    error_drag_array = np.std(all_drag_array,axis=0)

    plt.errorbar(averaged_Re_array,averaged_drag_array,
            xerr=error_Re_array,
            yerr=error_drag_array,fmt='o-',capsize=5,label = first_key)
    plt.legend()
    plt.xlabel('Re')
    plt.ylabel('Drag (gf)')

    return averaged_Re_array,averaged_drag_array,error_Re_array,error_drag_array

def plot_with_std(data_all):
    for k in data_all:
        U_array, Re_array, drag_array, e2_array = convert_data(k,data_all[k])
        plt.errorbar(Re_array,drag_array,e2_array,fmt='o',capsize=5,label=k)
        plt.xlabel('Re')
        plt.ylabel('Drag (gf)')
    plt.legend()