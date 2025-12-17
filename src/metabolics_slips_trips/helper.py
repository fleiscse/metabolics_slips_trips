import pandas as pd
import numpy as np
from scipy.interpolate import interp1d


def metabolic_cost(vo2, vco2):
    return (16.58 * vo2 + 4.15 * vco2)


def compare_before_after(data, perturbation_times, duration=15,  metric='mean'):
    
    before = []
    after = []

    for i in perturbation_times:
        start = data[i:].index[0]
       
        b =  data.loc[start-duration:start]
        a =  data.loc[start:start+duration]

        if metric=='mean':
            val_before = np.mean(b)
            val_after = np.mean(a)
        elif metric=='std':
            val_before = np.std(b)
            val_after = np.std(a)
        else:
            raise ValueError("metric not implemeted yet")

        before.append(val_before)
        after.append(val_after)
  
    return before, after

def all_responses(data, perturbation_times):
    new_time_min = np.round(np.arange(data.index.min(), data.index.max(), 0.01),3)
    f = interp1d(data.index, data)
    vo2_resampled_min = f(new_time_min)
    vo2_resampled_min = pd.DataFrame(vo2_resampled_min, index = new_time_min)

    all_ts = []
    for i in perturbation_times:
    
    #closest_value = vo2_resampled_f[i:].index[0]
    #diff = closest_value - i
        interval = vo2_resampled_min[i-9.999:i+45.001]
    
    
        inv = np.array(interval)
        if len(inv)>5500:
            inv = inv[:5500]
    #inv = [i[0] for i in inv] 
        all_ts.append(inv)
        
    responses  = np.array(all_ts)
    return responses

def average_response(responses):
    return np.mean(responses[:,:,0], axis=0)

# PROCESSING

def resample(data, dt=0.01):
    new_time_min = np.round(np.arange(data.index.min(), data.index.max(), dt),3)
    f = interp1d(data.index, data)
    vo2_resampled_min = f(new_time_min)
    vo2_resampled_min = pd.DataFrame(vo2_resampled_min, index = new_time_min)
    return vo2_resampled_min

    
def sliding_window(data, window_length=10, dt=0.01): 
    rolling_mean = data.rolling(window = int(window_length/dt), center=True).mean()
    return rolling_mean

def gaussian_sliding_window(data, sigma):
    rolling_mean_breaths_gaussian = gaussian_filter1d(vo2, sigma=0.1)
    return rolling_mean_breaths_gaussian

# FEATURE EXTRACTION 

#def baseline(data, perturbation_times, t=10):
 #   # we consider the t seconds before each perturbation as the baseline 
    

