import numpy as np
import mne
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.pyplot as plt

Data_folder = '/Users/bolger/Documents/work/Projects/Brain-IHM/Data_Agent'
Save_folder = '/Users/bolger/Documents/work/Projects/Brain-IHM/Data_Agent/EpochData_bis'
channels_txt = os.path.join(Data_folder, 'channels_ref.txt')
subject = 'S33'
block_name = ['S33-Film2'] #, 'S01-Film4'
subject_folder = os.path.join(Data_folder, subject)
curr_savepath = os.path.join(Save_folder, subject)
if os.path.exists(curr_savepath):
    print(f'The save directory {curr_savepath} already exists.\n')
else:
    print(f'The save directory {curr_savepath} does not exist. Creating...\n')
    os.makedirs(curr_savepath)

for counter, film in enumerate(block_name):
    fulldir_curr = os.path.join(subject_folder,film)
    fulldir_content = os.listdir(fulldir_curr)
    data_interp = [x for x in fulldir_content if x.endswith('ssinterp.set')]
    data_fullpath = os.path.join(fulldir_curr,data_interp[0])
    currdata = mne.io.read_raw_eeglab(data_fullpath, preload=True)
    channoms_curr = currdata.info['ch_names']
    print(channoms_curr)
    montage = mne.channels.make_standard_montage('standard_1020')
    currdata.set_montage(montage)
    srate = currdata.info['sfreq']
    cdata = currdata.get_data(picks='Cz')
    psd = mne.time_frequency.psd_array_welch(cdata, fmin=0, fmax=80, sfreq=srate)
    mne.viz.plot_raw_psd(currdata, fmin=0, fmax=50)

    mne.viz.plot_raw(currdata, duration=20, n_channels=64)
    events = mne.events_from_annotations(currdata)
    condnames_dict = events[1]
    condnames = [*condnames_dict]
    print(f'The conditions for current data-set are: {condnames}')

    # Epoch the continuous data
    currdata_epoched = mne.Epochs(currdata, events[0], event_id = events[1], tmin=-0.25, tmax=0.8, event_repeated='merge') # Baseline correction automatically carried out/
    mne.viz.plot_epochs(currdata_epoched, n_epochs=10, n_channels=64)
    currtitle = data_interp[0]
    split_title = currtitle.split("-")
    curr_eptitle = split_title[0]+'-epo.fif'
    curr_savefullpath = os.path.join(curr_savepath, curr_eptitle)
    currdata_epoched.save(curr_savefullpath, overwrite=True)

    currdata_epoched.plot()
