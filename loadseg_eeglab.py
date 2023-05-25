import numpy as np
import mne
import matplotlib.pyplot as plt
import glob
import os
import matplotlib.pyplot as plt

def align_zeros(axes):
    ylims_current = {}  # Current ylims
    ylims_mod = {}  # Modified ylims
    deltas = {}  # ymax - ymin for ylims_current
    ratios = {}  # ratio of the zero point within deltas

    for ax in axes:
        ylims_current[ax] = list(ax.get_ylim())
        # Need to convert a tuple to a list to manipulate elements.
        deltas[ax] = ylims_current[ax][1] - ylims_current[ax][0]
        ratios[ax] = -ylims_current[ax][0] / deltas[ax]

    for ax in axes:  # Loop through all axes to ensure each ax fits in others.
        ylims_mod[ax] = [np.nan, np.nan]  # Construct a blank list
        ylims_mod[ax][1] = max(deltas[ax] * (1 - np.array(list(ratios.values()))))
        # Choose the max value among (delta for ax)*(1-ratios),
        # and apply it to ymax for ax
        ylims_mod[ax][0] = min(-deltas[ax] * np.array(list(ratios.values())))
        # Do the same for ymin
        ax.set_ylim(tuple(ylims_mod[ax]))

# Load in the segmented data.
epochData_folder = '/Users/bolger/Documents/work/Projects/Brain-IHM/Segmented_Data_Jan2023/Agent/incongruent/'
conds = ['CongruBL', 'InCongruBL']
incong_path = os.path.join(epochData_folder, conds[1])
cong_path   = os.path.join(epochData_folder, conds[0])

incongarr = os.listdir(incong_path)
congarr = os.listdir(cong_path)

incongfiles = [x for x in incongarr if x.endswith('.set')]
congfiles = [x for x in congarr if x.endswith('.set')]

congfiles_full = [os.path.join(cong_path, f) for f in congfiles]
incongfiles_full = [os.path.join(incong_path, f) for f in incongfiles]

cnt1 = 0
for idx_cong in congfiles_full:
    currep = mne.read_epochs_eeglab(idx_cong)
    currtitle = idx_cong[:-4]+'-ave.fif'
    curr_evoked = currep['Verbal-cong/Gestual-cong'].average()
    curr_evoked_data = np.asarray(curr_evoked.get_data())
    mne.write_evokeds(currtitle, curr_evoked, overwrite=True)
    curr_size = np.shape(curr_evoked_data)
    print(curr_size)
    channoms_curr = currep.info['ch_names']
    cindx = channoms_curr.index('Fp1')
    print(cindx)
    print(len(channoms_curr))
    if cnt1==0:
        cong_cat = curr_evoked_data[0:64,0:538]
    else:
        cong_cat = np.stack((cong_cat, curr_evoked_data[0:64,0:538]), axis=2)
        cong_cat = np.mean(cong_cat, 2)
    cnt1+=1

cnt2=0
for idx_incong in incongfiles_full:
    currep = mne.read_epochs_eeglab(idx_incong)
    currtitle = idx_incong[:-4]+'-ave.fif'
    curr_evoked = currep.average()
    curr_evoked_data = np.asarray(curr_evoked.get_data())
    mne.write_evokeds(currtitle, curr_evoked, overwrite=True)
    curr_size = np.shape(curr_evoked_data)
    channoms_curr = currep.info['ch_names']
    cindx = channoms_curr.index('Fp1')
    curr_size = np.shape(curr_evoked_data)
    print(curr_size)
    print(cindx)
    print(len(channoms_curr))
    if cnt2==0:
        incong_cat = curr_evoked_data[0:64,0:538]
    else:
        incong_cat = np.stack((incong_cat, curr_evoked_data[0:64,0:538]), axis=2)
        incong_cat = np.mean(incong_cat, 2)
    cnt2+=1

times = currep.times
channoms = currep.info['ch_names']
roi = ['F3', 'Fz', 'F4', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4',
       'P3', 'Pz', 'P4']
eindx = [channoms.index(elabels) for elabels in roi]

fig = plt.figure()

for counter in range(0,12):

    ax1 = plt.subplot(4,3,counter+1, frameon=False)
    ax1.plot(times[0:538], cong_cat[eindx[counter]-10,:], 'b-', label='Human-incongru')
    ax1.plot(times[0:538], incong_cat[eindx[counter]-10,:], 'r-', label='Agent-incongru')
    plt.axvline(x=0, c="black")
    plt.axhline(y=0, c="black")
    ax1.set_xlabel('time (seconds)')
    ax1.set_title(roi[counter])
    ax1.invert_yaxis()

plt.legend()
plt.show()



