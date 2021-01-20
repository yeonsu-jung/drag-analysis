# %%
import nidaqmx
from nidaqmx import constants
from nidaqmx import stream_readers
from nidaqmx import stream_writers
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
from shutil import copyfile
import datetime
# %%
def set_motor_speed(motor_speed=None):
    if motor_speed is None:
        answer = input('Enter motor speed: ')    
        motor_speed = int(answer)    
        return 'Whatever'      
    
    with nidaqmx.Task() as write_task:
        write_task.ao_channels.add_ao_voltage_chan("Dev1/ao0")        
        MOTOR_SPEED_VOLT = motor_speed/6 # 60 Hz = 10 Volts
        write_task.write( MOTOR_SPEED_VOLT )    


def run_calibration(path_in,time_span = 10):    
    print('=== Load cell calibration procedure ===')

    while True:
        try:
            weight = input('Enter the weight: ')
            weight = float(weight)        
            break
        except ValueError:
            print("Please try again ...")

    sampling_frequency = 20000 #sample frequency        
    number_of_samples = time_span * sampling_frequency
    
    plt.draw()

    with nidaqmx.Task() as read_task:
        read_task.ai_channels.add_ai_voltage_chan("Dev1/ai0")    
        
        read_array = np.zeros((number_of_samples),dtype=np.float64)

        read_task.timing.cfg_samp_clk_timing(rate=sampling_frequency,
                                    active_edge=constants.Edge.RISING,
                                    sample_mode=constants.AcquisitionType.CONTINUOUS, 
                                    samps_per_chan=number_of_samples)

        reader = stream_readers.AnalogSingleChannelReader(read_task.in_stream)       

        t = time.time()
        print("Reading start")
        reader.read_many_sample(read_array, number_of_samples_per_channel=number_of_samples,timeout=1000)
        elapsed = time.time() - t
        print("Reading done. Elapsed time: %.2f sec." %elapsed)
        np.savetxt(os.path.join(path_in, '%.2f.csv' %weight), read_array.T,delimiter=",")

        read_task.stop()                    

    plt.plot(read_array)
    plt.show()

def calibrate_data(path_in):
    csv_list = [x for x in os.listdir(path_in) if x not in ['_result.txt']]

    result_path = os.path.join(path_in,'_result.txt')
    if os.path.isfile(result_path):
        df = pd.read_csv(result_path,comment='#',header=None)    
        result = df.to_numpy().flatten()

    def read_calibration_data(path):
        df = pd.read_csv(path,nrows=1,comment='#',header=None)
        if isinstance(df.iloc[0,0], str):   
            df = pd.read_csv(os.path.join(path_in,file_name),skiprows=4,names=['time','data'],comment='#')
            df = df.data.to_numpy()
        else:
            df = pd.read_csv(os.path.join(path_in,file_name),names=['data']).to_numpy()
        return df

    def linear_function(x,a,b):
        return a*x + b

    N = len(csv_list)
    weight_array = np.array([])
    data_array = np.array([])

    plt.figure()
    for file_name in csv_list:    
        weight, rhs = file_name.split(".csv", 1)
        weight = float(weight)    
        data = np.mean(read_calibration_data(os.path.join(path_in, file_name)))
        plt.scatter(weight,data)
        weight_array = np.append(weight_array,weight)
        data_array = np.append(data_array,data)

    plt.plot(weight_array,data_array,'o')
    popt,pcov = curve_fit(linear_function,weight_array,data_array)
    xx = np.linspace(0,50,100)
    plt.plot(xx,linear_function(xx,*popt))
    plt.title('y = %.4f x + %.4f' %(popt[0],popt[1]))
    perr = np.sqrt(np.diag(pcov))

    print(perr)
    print('a = %.6f +- %.6f' %(popt[0],perr[0]))

    result = np.array([popt[0],perr[0]])
    np.savetxt(os.path.join(path_in, '_result.txt'),result,delimiter=',')

def make_folder(path_in):        
    if os.path.isdir(path_in):
        while True:
            y_or_n = input('Directory already exists. Do you want to overwrite? (y/n)')
            if y_or_n == 'y':
                break
            elif y_or_n == 'n':
                sys.exit(1)
            else:
                print('Please enter y or n')
    else:
        os.mkdir(path_in)
        print('Successfully created the directory: %s' %path_in)



def run_experiment_set(path_in,motor_speed_array,samp_freq_array,settle_time_array,t_span_array,param_string):
    print('=== Starting drag experiment set ===')    

    # while True:
    #     try:
    #         param_string = input('Enter the parameter string (#_#_#): ')
    #         break
    #     except ValueError:
    #         print("Please try again ...")

    folder_path = os.path.join(path_in, param_string)   
    
    if not os.path.isfile(os.path.join(path_in,param_string,'_zero.csv')):
        print('No zeroing result found. Do zeroing first')
        sys.exit(1)
    elif len(os.listdir(folder_path)) > 1:
        folder_path = duplicate_folder(folder_path)       

    tt = time.localtime()
    start_time = time.strftime("%Y-%m-%d_%H:%M:%S", tt)
    print("Started at %s" %start_time)

    # length check
    try:
        np.vstack((samp_freq_array,settle_time_array,t_span_array,motor_speed_array))
    except:
        print('Use same size arrays.') 
        sys.exit(1)

    estimated_time = (np.sum(settle_time_array)+np.sum(t_span_array))/60
    print("Estimated time to complete: %.2f min" %estimated_time)

    i = 0
    for motor_speed in motor_speed_array:
        file_name = "%.2f.csv"%(motor_speed)       
        file_path = os.path.join(folder_path,file_name)        
        run_single_experiment(motor_speed,samp_freq_array[i],settle_time_array[i],t_span_array[i],export_path = file_path)
        i = i + 1
    
    tt = time.localtime()
    end_time = time.strftime("%Y-%m-%d_%H:%M:%S", tt)
    print("Finished at %s" %end_time)


    metadata = ['Start time: %s' %start_time,
            'End time:  %s' %end_time,
            'Sampling frequency (Hz): %s' %np.array2string(samp_freq_array,precision=0),
            'Settling time (sec): %s' %np.array2string(settle_time_array,precision=0),
            'Reading span (sec): %s' %np.array2string(t_span_array,precision=0),
            'Motor speed (Hz): %s' %np.array2string(motor_speed_array,precision=2)]

    with open( os.path.join(folder_path, '_metadata.txt'), 'w') as f:
        for s in metadata:
            f.write(s + '\n')

    turn_off()


def run_single_experiment(motor_speed,samp_freq=20000,settle_time=30,t_span = 25,export_path = '_test.csv'):    
    SAMP_FREQ = samp_freq
    NO_SAMPLES = int(samp_freq * t_span)
    MOTOR_SPEED_VOLT = motor_speed/6 # 60 Hz = 10 Volts
    SETTLING_TIME = settle_time

    if MOTOR_SPEED_VOLT > 6:
        print("Voltage output must be lower than 6 Volts")
        sys.exit(1)

    with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task:
        read_task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        write_task.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        read_array = np.zeros(NO_SAMPLES,dtype=np.float64)

        read_task.timing.cfg_samp_clk_timing(rate=SAMP_FREQ,
                                active_edge=constants.Edge.RISING,
                                sample_mode=constants.AcquisitionType.CONTINUOUS, 
                                samps_per_chan = NO_SAMPLES)

        reader = stream_readers.AnalogSingleChannelReader(read_task.in_stream)       

        write_task.write( MOTOR_SPEED_VOLT )    
        print("Motor speed: %.2f Hz, Wait until settling..." %motor_speed)
        time.sleep(SETTLING_TIME)

        t = time.time()
        print("Reading start")

        tt = time.localtime()
        START_TIME = time.strftime("%Y-%m-%d_%H:%M:%S", tt)

        reader.read_many_sample(read_array, number_of_samples_per_channel=NO_SAMPLES,timeout=1000)

        tt = time.localtime()
        END_TIME = time.strftime("%Y-%m-%d_%H:%M:%S", tt)

        elapsed = time.time() - t
        print("Reading done. Elapsed time: %.2f sec." %elapsed)                        

        with open( export_path, 'w') as f:
            f.write('# %s\n'%START_TIME)
            f.write('# %s\n'%END_TIME)

        with open( export_path, "ab") as f:
            np.savetxt(f, read_array.T,delimiter=",")

        read_task.stop()

def turn_off():
    print('Turning off the motor...')
    with nidaqmx.Task() as write_task:
        write_task.ao_channels.add_ao_voltage_chan("Dev1/ao0")
        
        write_task.write(2)
        time.sleep(10)

        write_task.write(1)
        time.sleep(10)

        write_task.write(0)
        time.sleep(10)

def zeroing(path_in,time_span=300):           
    print('=== Load cell zeroing procedure ===')
    print('Time span: %d sec'%time_span)

    while True:
        try:
            param_string = input('Enter the parameter string (#_#_#): ')
            break
        except ValueError:
            print("Please try again ...")
    folder_path = os.path.join(path_in, param_string)    
    file_path = os.path.join(folder_path,'_zero.csv')
    make_folder(folder_path)

    sampling_frequency = min(300000/time_span,20000) #sample frequency
    wait_until_settling = 1 # time interval between motor control and drag reading
    # time_span = 60 # time to acquire data
    number_of_samples = int(time_span * sampling_frequency)

    # file_path = path_in + '_zero_%s.csv' %param_string
    plt.draw()

    with nidaqmx.Task() as read_task:
        read_task.ai_channels.add_ai_voltage_chan("Dev1/ai0")    
        
        read_array = np.zeros((number_of_samples),dtype=np.float64)

        read_task.timing.cfg_samp_clk_timing(rate=sampling_frequency,
                                    active_edge=constants.Edge.RISING,
                                    sample_mode=constants.AcquisitionType.CONTINUOUS, 
                                    samps_per_chan=number_of_samples)

        reader = stream_readers.AnalogSingleChannelReader(read_task.in_stream)       

        t = time.time()
        print("Reading start")
        reader.read_many_sample(read_array, number_of_samples_per_channel=number_of_samples,timeout=1000)
        elapsed = time.time() - t
        print("Reading done. Elapsed time: %.2f sec." %elapsed)
        np.savetxt(file_path, read_array.T,delimiter=",")    

        read_task.stop()
    
    plt.plot(read_array)    
    plt.title('Mean value = %.3f' %check_average_value(file_path))
    plt.show()

def duplicate_folder(path_in):
    path_handle = Path(path_in)
    
    name_in = path_handle.name
    existing_folders = [x for x in os.listdir(path_handle.parent) if name_in in x]
    
    no = 1
    while True:
        try:
            no = no + 1
            new_folder_path = path_in + ' (%d)'%(no)
            os.mkdir(new_folder_path)
            src = os.path.join(path_in,'_zero.csv')
            break
        except:
            pass   
    
    dst = os.path.join(new_folder_path,'_zero.csv')
    copyfile(src, dst)
    return new_folder_path


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

def three_seconds_law(x,a):
    return a*x**(3/2)

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

# def plot_data_in_folders(path_in):
#     exp_list = os.listdir(path_in)
#     for exp in exp_list:
        
# def open_and_plot(exp_path_in):


# %%
# duplicate_folder('data/2021-01-10/1_1_0')


# # %%
# # make_folder('calibration')
# folder_path = 'calibration/2021-01-10/'
# # make_folder(folder_path)
# run_calibration(folder_path)
# # %%
# folder_path = 'calibration/2021-01-10/'
# calibrate_data(folder_path)

# # %%
# # make_folder('zeroes')
# zeroing('zeroes',time_span = 300)

# # %%
# check_average_value('zeroes/_zero_2_1.5_1.5 (300 sec).csv')
# # %%
# check_average_value('zeroes\_zero_with_sample_in_water (120 sec).csv')

# # %%
# folder_path = 'data/2021-01-10/2_1.5_1.5 (3)'
# # make_folder(folder_path)
# N = 15
# motor_speed_array = np.hstack((np.array([0, 1]), np.linspace(1,27,N-1)))
# sampling_freq_array = np.ones(N+1)*10000
# settle_time_array = np.ones(N+1)*30
# t_span_array = np.ones(N+1)*50

# run_experiment_set(folder_path,motor_speed_array,sampling_freq_array,settle_time_array,t_span_array)

# %%