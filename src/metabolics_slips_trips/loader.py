import os
import glob
from tpcp import Dataset
from typing import List, Optional, Union
from itertools import   product
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from scipy import signal
import json 
from scipy.io import loadmat

def get_all_participant_IDs(main_folder):
    participant_folders = []

    # Traverse through all subdirectories and collect participant folder names
    for root, dirs, _ in os.walk(main_folder):
        for folder in dirs:
            participant_folders.append(folder)

    return participant_folders


from scipy import signal
from functools import lru_cache

@lru_cache(maxsize=1)
def _extract_file_metadata(base_path):
    final_files = {}

    p = os.path.join(base_path, "Met_Cost_Exp.xlsx")
    configs = pd.read_excel(p, index_col = 0, sheet_name=None)
    

    

    for p in sorted(base_path.rglob("*.mat")):
        *_, pat_id, trial = p.parts
        trial = trial.split(".")[0]
        trial_number = trial.strip('trial')
        magnitude = configs[pat_id].magnitude.loc[int(trial_number)]
        

        final_files[(pat_id, trial, magnitude)] = p


    return final_files


class DatasetVo2(Dataset):
    def __init__(self, data_path,
                 use_lru_cache: bool = True,
                 groupby_cols: Optional[Union[List[str], str]] = None,
                 subset_index: Optional[pd.DataFrame] = None,
                 ):
        self.data_path = data_path
        self.use_lru_cache = use_lru_cache

        p = os.path.join(data_path, "Met_Cost_Exp.xlsx")
        self.configs = pd.read_excel(p, index_col = 0, sheet_name=None)

        super().__init__(groupby_cols=groupby_cols, subset_index=subset_index)


    @property
    def _file_metadata(self):
        return _extract_file_metadata(Path(self.data_path))

  
    @property
    def sync_delay(self):
        idx = self.index['id'][0]
        with open(os.path.join(self.data_path, "delays.txt")) as f:
            d = json.load(f)
        return d[idx]
    

    def _get_file_from_metadata(self, metdata_tuple):
        return self._file_metadata[metdata_tuple]


    def create_index(self):
        final_files = pd.DataFrame(list(self._file_metadata.keys()), columns=["id", "trial", "magnitude"])
        return final_files


    #----------functions related to LOADING the data-----------------------

    
    @property
    def vo2_data(self):
        #file_path = self._get_file_from_metadata(self.index['id'][0])
        p_path = os.path.join(self.data_path, self.index['id'][0])

        file = glob.glob(os.path.join(p_path, "*.XLS"))[0]
        columns = pd.read_excel(file, engine="xlrd", skiprows = 26, nrows = 2).columns
        columns = [i.strip() for i in columns]
        data = pd.read_excel(file, engine="xlrd", skiprows = 30, header=None)
        data.columns = columns
        data.TIME = data.TIME * 60

        data.set_index('TIME', inplace=True)

        data = data.dropna()
        data.index = [round(t, 2) for t in data.index]

        trial_start, trial_end = self.trial_start_end     
    
        return data[trial_start:trial_end]
        

    @property
    def grf_data(self): 

        key_tuple = (
        self.index["id"][0],
        self.index["trial"][0],
        self.index["magnitude"][0],
        )

        file = self._get_file_from_metadata(key_tuple)
        pdata = loadmat(file)
        key = [k for k in pdata.keys() if not k.startswith("__")][0]
        fyl = pdata[key][0][0][6][0][0][5][2]
        fyr = pdata[key][0][0][6][0][1][5][2]

        d = pd.DataFrame([fyl, fyr]).T
        grf = self.filter_data(d, order=2, fc= 10, fs=1000)
        grf.index = grf.index / 1000
        grf.columns = ["fy_left", "fy_right"]
        #grf.index = grf.index / 1000

        return grf

    def filter_data(self, data, order=3, fc= 0.04, fs=100):
        b, a = signal.butter(order, fc, fs = fs)
        column_names = data.columns
        index = data.index
        filtered_data = signal.filtfilt(b, a, data, axis=0)
        filtered_data = pd.DataFrame(filtered_data, columns=column_names)
        filtered_data.index = index
        return filtered_data

    @property
    def trial_start_end(self):
        #assume trial starts ~1min before and ends ~1min after first and last perturbation 
        impulse_times = self.impulse_indices #these are the impulse times rel. to the start of the qualisys recording! But we need to know add the time difference bteween qualisys and the met cart!
        start, end = (impulse_times[0]/1000)-60, (impulse_times[-1]/1000)+60 #convert from index to seconds
        pat_config = self.configs[self.index["id"][0]] #this gets the correct sheet of the excel table
        trial_number = self.index["trial"][0].strip("trial")
        delta_t = pat_config.delta_t_qualisys_met.loc[int(trial_number)]
        
        return start+delta_t, end+delta_t

    @property
    def impulse_indices(self):
        key_tuple = (
        self.index["id"][0],
        self.index["trial"][0],
        self.index["magnitude"][0],
        )

        file = self._get_file_from_metadata(key_tuple)
        pdata = loadmat(file)
        key = [k for k in pdata.keys() if not k.startswith("__")][0]

        impulse = pdata[key][0][0][5][0][0][-1][28,:]
       

        impulse_data = pd.DataFrame(impulse).T
        impulse_data.index = np.arange(len(impulse_data))/1000
        imp = np.array(impulse_data)[0]
        indices = np.where(imp > 3)[0]
        breaks = np.where(np.diff(indices) != 1)[0] + 1

        impulse_starts = indices[np.insert(breaks, 0, 0)]

      
     
     
        return impulse_starts

    @property
    def impulse_times(self):
        impulse_indices = self.impulse_indices 
        
        pat_config = self.configs[self.index["id"][0]] #this gets the correct sheet of the excel table
        trial_number = self.index["trial"][0].strip("trial")
        delta_t = pat_config.delta_t_qualisys_met.loc[int(trial_number)]

        
        impulse_times = [np.round(i/1000 + delta_t, 3) for i in impulse_indices]
       
        return impulse_times

    @property
    def hs_after_perturbation_times(self):
        impulse_indices = self.impulse_indices 
        impulse_times = [np.round(i/1000, 3) for i in impulse_indices]
        
        grf = self.grf_data
        fyl = grf.iloc[:,0]
        fyr = grf.iloc[:,1]
        fy1 = fyl[np.round(np.array(impulse_times) + 0.3, 3)] #0.3 s after the impulse start the leg which is perturbed should be in swing
        fy2 = fyr[np.round(np.array(impulse_times) + 0.3, 3)]

        left_hs = []
        for i in fy1.index[fy1<50]:
            next_range = fyl[i:i+3] #looks at the next 3 seconds
            hs = next_range.index[next_range>50][0] #first time where the GRF exceeds 50N = HS
            left_hs.append(hs)
        right_hs = []
        for i in fy2.index[fy2<50]:
            next_range = fyr[i:i+3]
            hs = next_range.index[next_range>50][0]
            right_hs.append(hs)

        right_hs.extend(left_hs)
        pat_config = self.configs[self.index["id"][0]] #this gets the correct sheet of the excel table
        trial_number = self.index["trial"][0].strip("trial")
        delta_t = pat_config.delta_t_qualisys_met.loc[int(trial_number)]
        all_hs = [np.round(i+delta_t, 3) for i in right_hs]
        
        return all_hs


    @property
    def trial_times(self):
        idx = self.index['id'][0]
        
        met_data_files = glob.glob(f"../data/{idx}/*trials.xlsx")
        path = met_data_files[0]
        p = pd.read_excel(path)
        p.columns = [d.strip() for d in p.columns]
        trial_times = {}

        for i in range(len(p)):
            trial_times[i+1] = [round(p['Trial Start'][i]*60, 2), round(p['Trial End'][i]*60,2)]
        return trial_times

    @property
    def magnitudes(self):
        idx = self.index['id'][0]
        
        met_data_files = glob.glob(f"../data/{idx}/*trials.xlsx")
        path = met_data_files[0]
        p = pd.read_excel(path)
        p.columns = [d.strip() for d in p.columns]
        return p['Ptb Random']
       

  
    
   

    def resample(self, index, vo2, dt_new = 0.001):
        
        new_time_min = np.round(np.arange(index.min(), index.max(), dt_new),3)
        f = interp1d(index, vo2)
        vo2_resampled_min = f(new_time_min)
        vo2_resampled_min = pd.DataFrame(vo2_resampled_min, index = new_time_min)
        return vo2_resampled_min

