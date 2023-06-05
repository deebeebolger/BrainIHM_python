import numpy as np
import mne
import matplotlib.pyplot as plt
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import SpanSelector
matplotlib.use('TkAgg')

def sem_calc(evokedIn):
   """
   Function to calculate the standard error of the mean for each channel taking into account all subjects.
   :param evokedIn: the evoked data for each subject
   :return: the upper and lower limit of sem to plot
   """
   evshape = np.shape(evokedIn)
   sem_sujs = np.std(evokedIn, 2, ddof=1)/np.sqrt(evshape[2])
   sujmean = np.mean(evokedIn,2)
   upperlim = sujmean + sem_sujs
   lowlim   = sujmean - sem_sujs

   return upperlim, lowlim

def onselect(xmin, xmax):
    indmin, indmax = np.searchsorted(x, (xmin, xmax))
    indmax = min(len(times) - 1, indmax)

    region_x = times[indmin:indmax]
    region_y1 = y1[indmin:indmax]
    region_y2 = y2[indmin:indmax]
    print(region_x)
    return region_x, region_y1, region_y2



dir_base = '/Users/bolger/Documents/work/Projects/Brain-IHM'
Groups = ['Human', 'Human']
# Define the video-type and the feedback types.
Conds2plot = ['Congru-Congru', 'InCongru-Congru']  # Needs to be the same length as groups.

# Initialize the data.
EvokedAll_cond = []
EvokedAll = []
condnames = []
Lims_upper = []
Lims_lower = []

for gcount, gcurr in enumerate(Groups):
    Group_dir = 'Data_' + gcurr
    Group_path = os.path.join(dir_base, Group_dir, 'EpochData')
    GPath_contents = os.listdir(Group_path)
    datadir_noms = [x for xind, x in enumerate(GPath_contents) if x.startswith('S')] # Extract names of the data folders.
    condcurr = Conds2plot[gcount]
    condcurrf = gcurr + '-' + condcurr
    condnames.append(condcurrf)
    EvokedLoad = {}

    for scount, scurr in enumerate(datadir_noms):
        Sujpath_curr = os.path.join(Group_path, scurr)
        sujcurr_contents = os.listdir(Sujpath_curr)
        evoked2load = [x1 for x1 in sujcurr_contents if condcurrf in x1]

        if len(evoked2load)>0:
            print(f' Loading the evoked file: {evoked2load[0]}')
            evoked2load_path = os.path.join(Sujpath_curr, evoked2load[0])
            condsplit = condcurr.split('-')
            cond2load = condsplit[1]
            EvokedLoad[scount] = mne.read_evokeds(evoked2load_path,condition=cond2load, baseline=None, kind='average')
            Evoked_data = EvokedLoad[scount].get_data()

            if scount == 0:
                evokeddata_cat = Evoked_data
            elif scount > 0:
                evokeddata_cat = np.dstack((evokeddata_cat, Evoked_data))

        elif len(evoked2load)==0:
            print(f'The condition file {condcurr} does not exist for {scurr}, {gcurr}.')



    Evokeddata_mean = np.average(evokeddata_cat,2)
    EvokedAll.append(EvokedLoad)

    if gcount == 0:
        channoms  = EvokedLoad[scount].info['ch_names']
        times = EvokedLoad[scount].times
        EvokedAll_cond = Evokeddata_mean

    elif gcount>0:
        EvokedAll_cond = np.dstack((EvokedAll_cond, Evokeddata_mean))

    limup, limlow = sem_calc(evokeddata_cat)  # Should output standard deviation from mean for each channel.
    Lims_upper.append(limup)
    Lims_lower.append(limlow)


## Plot the ERP data for pre-defined electrodes
roi = ['F3', 'Fz', 'F4', 'FC3', 'FCz', 'FC4', 'C3', 'Cz', 'C4',
       'P3', 'Pz', 'P4']
eindx = [channoms.index(elabels) for elabels in roi]
shape_ev = np.shape(EvokedAll_cond)

rows = 4
cols = 3
fig, axes = plt.subplots(nrows=rows,ncols=cols, figsize=(20,20))
counter = 0

for axs, ecurr in zip(axes.ravel(), eindx):

    upper1 = Lims_upper[0]
    upper2 = Lims_upper[1]

    lower1 = Lims_lower[0]
    lower2 = Lims_lower[1]

    y1 = EvokedAll_cond[ecurr, :, 0]
    y2 = EvokedAll_cond[ecurr, :, 1]
    x = times
    axs.plot(times, EvokedAll_cond[ecurr, :, 0], 'b-', label=condnames[0])
    axs.plot(times, upper1[ecurr, :], 'b-')
    axs.plot(times, lower1[ecurr, :], 'b-')
    axs.fill_between(times, lower1[ecurr, :], upper1[ecurr,:])

    axs.plot(times, EvokedAll_cond[ecurr, :, 1], 'r-', label=condnames[1])
    axs.plot(times, upper2[ecurr, :], 'r-')
    axs.plot(times, lower2[ecurr, :], 'r-')
    axs.fill_between(times, lower2[ecurr, :], upper2[ecurr, :])
    axs.axvline(x=0, c="black")
    axs.axhline(y=0, c="black")
    axs.set_title(channoms[ecurr])
    axs.invert_yaxis()
    axs.set_frame_on(0)
    axs.set_ylim(bottom=6*(pow(10, -6)), top=-3*(pow(10, -6)))
    if counter<= ((rows*cols)-cols)-1:
        axs.set_xlabel(' ')
        axs.get_xaxis().set_ticks([])
    elif counter>((rows*cols)-cols)-1:
        axs.set_xlabel('time (seconds)')

    span = SpanSelector(
        axs,
        onselect,
        "horizontal",
        useblit=True,
        props=dict(alpha=0.5, facecolor="tab:blue"),
        interactive=True,
        drag_from_anywhere=True
    )

    counter+=1

plt.legend()
plt.show()


# Add interactive plots to select intervals and plot the topographies.
# This can be carried out using mne.viz.plot_topomap()
# The above function takes the data in array form..Perfect!!




#mne.viz.plot_topoplot()