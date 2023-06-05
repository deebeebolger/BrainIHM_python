import numpy as np
import mne
import matplotlib.pyplot as plt
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

dir_base = '/Users/bolger/Documents/work/Projects/Brain-IHM'
Group = ['Agent']
DataType = 'EpochData_bis'

eog_channels = ['EXG1', 'EXG2', 'EXG3', 'EXG4', 'EXG5', 'EXG6', 'EXG7', 'EXG8']
misc_channels = ['GSR1', 'GSR2']

for gcount, gcurr in enumerate(Group):

    gpath = os.path.join(dir_base, 'Data_'+Group[gcount])
    GroupPath = os.path.join(gpath, DataType)
    GPath_contents = os.listdir(GroupPath)
    datadir_noms = [x for xind, x in enumerate(GPath_contents) if x.startswith('S')] # Extract names of the data folders.

    for scount, sujs in enumerate(datadir_noms):

        sujcurr_dir = os.path.join(GroupPath, sujs)
        sujf = os.listdir(sujcurr_dir)
        sujfiles = [sfile for sfile in sujf if sfile.endswith('epo.fif')]

        for sujcnt, sujcurr in enumerate(sujfiles):

            print(f'The current dataset is {sujcurr}')
            current_suj = os.path.join(sujcurr_dir, sujcurr)
            print(f'The current sujet path is {current_suj}')
            EpochIn = mne.read_epochs(current_suj, preload=True)
            EpochBL = EpochIn.copy().apply_baseline()
            chans = EpochBL.ch_names

            # Drop the external channels.
            chan2drop = [elec for elec in chans if elec.startswith('GSR')]
            exg2drop = [exg for exg in chans if exg.startswith('EXG')]

            if len(chan2drop)>0:
                EpochBL.drop_channels(chan2drop)
            if len(exg2drop)>0:
                EpochBL.drop_channels(exg2drop)
            chans_eeg = EpochBL.ch_names
            chtype_map = {}
            for k1, key1 in enumerate(chans_eeg):
                chtype_map[key1] = 'eeg'
            EpochIn.set_channel_types(chtype_map)
            print(f"Epochs baseline: {EpochBL.baseline}")
            print(f"The number of channels is: {len(EpochBL.info['ch_names'])}")  # Should be 74

            # Plot the epoched data and mark noisy segments.
            eventid = EpochBL.event_id
            Events = EpochBL.events

            if gcurr == 'Agent':
                print(f'The current group is {gcurr}')
                K = eventid.keys()
                mcong = ['Gestual-cong/Verbal-cong', 'Verbal-cong']
                mincong = ['Gestual-incong/Verbal-incong', 'Verbal-incong']
                cong_m = [m for m in mcong if m in K ]
                incong_m = [mi for mi in mincong if mi in K]
                if len(mcong)>0:
                     eventid['Congru'] = eventid.pop(cong_m[0])
                     if len(incong_m)>0:
                         print(f' Both congruent and incongruent feedbacks in current dataset.')
                         eventid['InCongru'] = eventid.pop(incong_m[0])
                         EpochBL["Congru", "InCongru"].plot(block=True, picks=['eeg'], n_epochs=10, n_channels=len(chans_eeg),
                                                events=Events, event_id=eventid,
                                                butterfly=False, show=True)

                     elif len(incong_m)==0:
                         print('No incongruent feedbacks in current dataset.')
                         EpochBL["Congru"].plot(block=True, picks=['eeg'], n_epochs=10, n_channels=len(chans_eeg), events=Events, event_id=eventid,
                                        butterfly=False, show=True)
            elif gcurr == 'Human':
                EpochBL.plot(block=True, picks=['eeg'], n_epochs=10, n_channels=len(chans_eeg), events=Events,
                                       event_id=eventid,
                                       butterfly=False, show=True)

            # Determine the conditions in current data set.
            A= list((EpochBL.event_id.keys()))
            is_congru = [idx for idx, ic in enumerate(A) if ic == 'Congru']
            is_incongru = [idx2 for idx2, ic2 in enumerate(A) if ic2 == 'InCongru']

            if len(is_incongru) == 0:
                print('The current data set only contains Congruent feedbacks.')
                print(f'The current data set is {gcurr}: Congru')
                evoked_cong = EpochBL["Congru"].average()

                # Set up evoked file title to save evoked file.
                f2split = sujcurr.split('-')
                CongEv_fnom = f2split[0]+'-'+gcurr+'-Congru-Congru-ave.fif'
                ave_savepath = os.path.join(sujcurr_dir, CongEv_fnom)
                evoked_cong.save(ave_savepath, overwrite=True)

            elif len(is_incongru) >0:
                print('The current data set contains Congruent & Incongruent feedbacks')
                print(f'The current data set is {gcurr}: Congru & InCongru')
                evoked_cong_cong = EpochBL["Congru"].average()
                evoked_cong_incong = EpochBL["InCongru"].average()

                # Set up evoked file title to save evoked file.
                f2split = sujcurr.split('-')
                CongEv_fnom = f2split[0] + '-' + gcurr + '-InCongru-Congru-ave.fif'
                InCongEv_fnom = f2split[0] + '-' + gcurr + '-InCongru-InCongru-ave.fif'
                ave_savepath_cong = os.path.join(sujcurr_dir, CongEv_fnom)
                ave_savepath_incong = os.path.join(sujcurr_dir, InCongEv_fnom)
                evoked_cong_cong.save(ave_savepath_cong, overwrite=True)
                evoked_cong_incong.save(ave_savepath_incong, overwrite=True)
