"""
Dataset_Loader.py

A moddule that provides functions for the loading of three 
datasets: BCI competition IV dataset 2a, PhysioNet Motor Imagery dataset, and 
the MindRove Arc dataset.

"""
import mne
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar
from mne_faster import find_bad_channels, find_bad_epochs, find_bad_components, find_bad_channels_in_epochs
import pickle
import json
from collections import defaultdict

def BCICIV2a_Dataset_Loader(path_to_dataset, subjects, tmin = 0, tmax = 4, resample_fs = 128,
                            fmin =1, fmax = 45, pick_6 = True, CAR = True, filter = True, plot=False, plot_psd=False, do_faster = True,
                            pickle_dataset = False, pickle_path = None):
    """Creates a dataset for the subjects of the BCI competition IV 2a in the form of a 
    dictionary and in a within-subject approach where each subject's data and labels are seperate. 

    Args:
        path_to_dataset (str): Path to the folder where the raw files are stored which can be downloaded from https://www.bbci.de/competition/iv/
        subjects (list): List containing integers in the range of 1 to 9 representing the subjects of the dataset to be processed
        tmin (float, optional): Epoching starts tmin before the cue onset. Defaults to 0.
        tmax (int, optional): Epoching ends tmax after the cue onset. Defaults to 4.
        resample_fs (int, optional): The frequency the signals are resampled to. Defaults to 128.
        fmin (int, optional): The lower cutoff frequency of the 5th order Butterworth bandpass filter. Defaults to 1.
        fmax (int, optional): The upper cutoff frequency of the 5th order Butterworth bandpass filter. Defaults to 45.
        pick_6 (bool, optional): Whether to keep only the 6 electrodes or to keep all 22 EEG electrodes. Defaults to True.
        CAR (bool, optional): Whether to apply common average referencing. Defaults to True.
        filter (bool, optional): Whether to apply bandpass filtering. Defaults to True.
        plot (bool, optional): Whether to generate a time series plot of the electrodes data after epoching. Defaults to False.
        plot_psd (bool, optional): Whether to plot the power spectral density before and after filterring. Defaults to False.
        do_faster (bool, optional): Whether to apply the FASTER algorithm. Defaults to True.
        pickle_dataset (bool, optional): Whether to generate a pickle object of the dataset so it can be loaded back again. Defaults to False.
        pickle_path (str, optional): Path to the destination where the pickle dataset object will be saved. Defaults to None.

    Returns:
        dict: The keys are integers corrosponding to the subject of the dataset. The values are tuples, where the first element is the epoched data
        of the subject in the shape (number of epochs, number of electrodes, number of time points), and the second element is the labels for each epoch in the shape (number of epochs,)
    """
    rename_mapping = { 'EEG-Fz': 'Fz',
                        'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
                        'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1','EEG-Cz': 'Cz', 'EEG-7': 'C2','EEG-C4': 'C4', 'EEG-8': 'C6',
                        'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
                        'EEG-14': 'P1', 'EEG-Pz': 'Pz', 'EEG-15': 'P2',
                        'EEG-16': 'POz', 'EOG-left': 'EOG-left', 'EOG-central': 'EOG-central', 'EOG-right': 'EOG-right' }
    montage = mne.channels.make_standard_montage('standard_1020')
    mne.set_log_level(verbose='CRITICAL')
    BCICIV2a_Dataset = {}

    for l in Bar(f'Subjects',max=9).iter(subjects):        
        raw = mne.io.read_raw_gdf(input_fname=f'{path_to_dataset}//A0{l}T.gdf', eog = ['EOG-left', 'EOG-central', 'EOG-right'],preload=True)
        raw.rename_channels(rename_mapping)
        raw.set_montage(montage)
        raw.info['bads'] = []
        events = mne.events_from_annotations(raw)
        eeg_picks = mne.pick_types(raw.info,eeg=True,eog=False)

        for i in range(len(events[0])):
            if l==4:
                events[0][i][-1] = events[0][i][-1] - 4
            else:
                events[0][i][-1] = events[0][i][-1] - 6
        
        epochs = mne.Epochs(raw,events[0],event_id={'left':1,'right':2,'foot':3,'tongue':4},
                            tmin = tmin,tmax = tmax,preload=True, baseline=None)
        epochs = epochs.crop(tmin=tmin,tmax=tmax,include_tmax= False)
        
        if filter:  
            if plot_psd:
                unfilt_epoch_psd = epochs.compute_psd(remove_dc=True)
            epochs = epochs.filter(fmin,fmax,method='iir',iir_params=dict(order=5,ftype='butter'))
            if plot_psd:
                filt_epoch_psd = epochs.compute_psd(remove_dc=True)      
        if plot_psd:
            fig, ax = plt.subplots()
            filt_epoch_psd.plot(color='r',spatial_colors=False,axes = ax)
            unfilt_epoch_psd.plot(spatial_colors=False,axes = ax)
            plt.show()

        epoch_labels_with_rejected_trials = [raw[2] for raw in events[0] if raw[2] in [1,2,3,4,-5]]
        # example:
        # [-5, 1, 2, 3, -5, 1, 2, -5, 1, 2, 3, 4, ....]
        indices_of_rejected_plus1 = [(i+1) for i, x in enumerate(epoch_labels_with_rejected_trials) if x==-5]
        # indicies of the rejected epochs, the ones after the -5 event
        # [1, 5, 8]
        indices_to_be_removed = [(x-(i+1)) for i, x in enumerate(indices_of_rejected_plus1) ]
        # without the rejected trials the epoch labels would be:
        # [1, 2, 3, 1, 2, 1, 2, 3, 4, ....]
        # so we need to drop the [0, 3, 5] indicies from the epochs without the -5 event
        epochs.drop(indices_to_be_removed)

        if do_faster:
            epochs.info['bads'] = find_bad_channels(epochs,eeg_ref_corr=False)
            if len(epochs.info['bads']) > 0:
                epochs.interpolate_bads()
            
            bad_epochs = find_bad_epochs(epochs)
            if len(bad_epochs) > 0:
                epochs.drop(bad_epochs)

            ica = mne.preprocessing.ICA(0.99,method='infomax').fit(epochs, picks = eeg_picks)
            ica.exclude = find_bad_components(ica, epochs)
            ica.apply(epochs)

            bad_channels_per_epoch = find_bad_channels_in_epochs(epochs, eeg_ref_corr=False)
            for i, b in enumerate(bad_channels_per_epoch):
                if len(b) > 0:
                    ep = epochs[i]
                    ep.info['bads'] = b
                    ep.interpolate_bads() 
                    epochs._data[i, :, :] = ep._data[0, :, :]
        
        if CAR:
            epochs.set_eeg_reference('average')
        
        epochs.resample(resample_fs)
        if pick_6:
            epochs.pick(['C5','C3','C1','C2','C4','C6'])
        else:
            epochs.pick(eeg_picks)

        epochs.equalize_event_counts()
        
        if plot:
            epochs.plot(n_channels = 6,n_epochs=5,events=events[0],block=True,
                        event_id={'left':1,'right':2,'foot':3,'tongue':4}
                        ,scalings=dict(eeg = 20e-6))
        
        labels = epochs.events[:,-1]
        data = epochs.get_data()

        mean = np.mean(data,axis=(0,2),keepdims=True)
        std = np.std(data,axis=(0,2),keepdims=True)
        data_normalized = (data - mean) / std

        BCICIV2a_Dataset[l] = ([data_normalized,labels])

    if pickle_dataset:
        with open(pickle_path,"wb") as f:
            pickle.dump(BCICIV2a_Dataset, f)
    return BCICIV2a_Dataset

def BCICIV2a_Transfer_Dataset_Loader(path_to_dataset, subjects, tmin = 0, tmax = 4, resample_fs = 128,
                                fmin =1, fmax = 45, pick_6 = True,
                                CAR = True, filter = True, plot=False, plot_psd=False, do_faster = True,
                                pickle_dataset = False, pickle_path = None):
    """Creates a concatenated dataset for the subjects of the BCI competition IV 2a in the form of a 
    tuple and in a cross-subject. 

    Args:
        path_to_dataset (str): Path to the folder where the raw files are stored which can be downloaded from https://www.bbci.de/competition/iv/
        subjects (list): List containing integers in the range of 1 to 9 representing the subjects of the dataset to be processed
        tmin (float, optional): Epoching starts tmin before the cue onset. Defaults to 0.
        tmax (int, optional): Epoching ends tmax after the cue onset. Defaults to 4.
        resample_fs (int, optional): The frequency the signals are resampled to. Defaults to 128.
        fmin (int, optional): The lower cutoff frequency of the 5th order Butterworth bandpass filter. Defaults to 1.
        fmax (int, optional): The upper cutoff frequency of the 5th order Butterworth bandpass filter. Defaults to 45.
        pick_6 (bool, optional): Whether to keep only the 6 electrodes or to keep all 22 EEG electrodes. Defaults to True.
        CAR (bool, optional): Whether to apply common average referencing. Defaults to True.
        filter (bool, optional): Whether to apply bandpass filtering. Defaults to True.
        plot (bool, optional): Whether to generate a time series plot of the electrodes data after epoching. Defaults to False.
        plot_psd (bool, optional): Whether to plot the power spectral density before and after filterring. Defaults to False.
        do_faster (bool, optional): Whether to apply the FASTER algorithm. Defaults to True.
        pickle_dataset (bool, optional): Whether to generate a pickle object of the dataset so it can be loaded back again. Defaults to False.
        pickle_path (str, optional): Path to the destination where the pickle dataset object will be saved. Defaults to None.

    Returns:
        tuple: The first element is the epoched data
        of the concatenated subjects in the shape (number of epochs, number of electrodes, number of time points), and the second element is the labels for each epoch in the shape (number of epochs,)
    """
    
    rename_mapping = { 'EEG-Fz': 'Fz',
                        'EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 'EEG-3': 'FC2', 'EEG-4': 'FC4',
                        'EEG-5': 'C5', 'EEG-C3': 'C3', 'EEG-6': 'C1','EEG-Cz': 'Cz', 'EEG-7': 'C2','EEG-C4': 'C4', 'EEG-8': 'C6',
                        'EEG-9': 'CP3', 'EEG-10': 'CP1', 'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4',
                        'EEG-14': 'P1', 'EEG-Pz': 'Pz', 'EEG-15': 'P2',
                        'EEG-16': 'POz', 'EOG-left': 'EOG-left', 'EOG-central': 'EOG-central', 'EOG-right': 'EOG-right' }
    montage = mne.channels.make_standard_montage('standard_1020')
    mne.set_log_level(verbose='CRITICAL')
    BCICIV2a_Dataset_epochs = []

    for l in Bar(f'Subjects',max=100).iter(subjects):
        raw = mne.io.read_raw_gdf(input_fname=f'{path_to_dataset}//A0{l}T.gdf', eog = ['EOG-left', 'EOG-central', 'EOG-right'],preload=True)
        raw.rename_channels(rename_mapping)
        raw.set_montage(montage)
        raw.info['bads'] = []
        events = mne.events_from_annotations(raw)
        eeg_picks = mne.pick_types(raw.info,eeg=True,eog=False)

        for i in range(len(events[0])):
            if l==4:
                events[0][i][-1] = events[0][i][-1] - 4
            else:
                events[0][i][-1] = events[0][i][-1] - 6

        epochs = mne.Epochs(raw,events[0],event_id={'left':1,'right':2,'foot':3,'tongue':4},
                            tmin = tmin,tmax = tmax,preload=True, baseline=None)
        epochs = epochs.crop(tmin=tmin,tmax=tmax,include_tmax= False)

        if filter:  
            if plot_psd:
                unfilt_epoch_psd = epochs.compute_psd(remove_dc=True)
            epochs = epochs.filter(fmin,fmax,method='iir',iir_params=dict(order=5,ftype='butter'))
            if plot_psd:
                filt_epoch_psd = epochs.compute_psd(remove_dc=True)
        if plot_psd:
            fig, ax = plt.subplots()
            filt_epoch_psd.plot(color='r',spatial_colors=False,axes = ax)
            unfilt_epoch_psd.plot(spatial_colors=False,axes = ax)
            plt.show()
        
        epoch_labels_with_rejected_trials = [raw[2] for raw in events[0] if raw[2] in [1,2,3,4,-5]]
        # example:
        # [-5, 1, 2, 3, -5, 1, 2, -5, 1, 2, 3, 4, ....]
        indices_of_rejected_plus1 = [(i+1) for i, x in enumerate(epoch_labels_with_rejected_trials) if x==-5]
        # indicies of the rejected epochs, the ones after the -5 event
        # [1, 5, 8]
        indices_to_be_removed = [(x-(i+1)) for i, x in enumerate(indices_of_rejected_plus1) ]
        # without the rejected trials the epoch labels would be:
        # [1, 2, 3, 1, 2, 1, 2, 3, 4, ....]
        # so we need to drop the [0, 3, 5] indicies from the epochs without the -5 event

        epochs.drop(indices_to_be_removed)

        if do_faster:
            epochs.info['bads'] = find_bad_channels(epochs,eeg_ref_corr=False)
            if len(epochs.info['bads']) > 0:
                epochs.interpolate_bads()
            
            bad_epochs = find_bad_epochs(epochs)
            if len(bad_epochs) > 0:
                epochs.drop(bad_epochs)

            
            ica = mne.preprocessing.ICA(0.99,method='infomax').fit(epochs, picks = eeg_picks)
            ica.exclude = find_bad_components(ica, epochs)
            ica.apply(epochs)

            bad_channels_per_epoch = find_bad_channels_in_epochs(epochs, eeg_ref_corr=False)
            for i, b in enumerate(bad_channels_per_epoch):
                if len(b) > 0:
                    ep = epochs[i]
                    ep.info['bads'] = b
                    ep.interpolate_bads() 
                    epochs._data[i, :, :] = ep._data[0, :, :]
        
        if CAR:
            epochs.set_eeg_reference('average')
        
        epochs.resample(resample_fs)

        if pick_6:
            epochs.pick(['C5','C3','C1','C2','C4','C6'])
        else:
            epochs.pick(eeg_picks)
        
        if plot:
            epochs.plot(n_channels = 6,n_epochs=5,events=events[0],block=True,
                        event_id={'left':1,'right':2,'foot':3,'tongue':4}
                        ,scalings=dict(eeg = 20e-6))

        BCICIV2a_Dataset_epochs.append(epochs)

    transfer_learning_epochs = mne.concatenate_epochs(BCICIV2a_Dataset_epochs)
    transfer_learning_epochs.equalize_event_counts()

    labels = transfer_learning_epochs.events[:,-1]
    data = transfer_learning_epochs.get_data() 
    mean = np.mean(data,axis=(0,2),keepdims=True)
    std = np.std(data,axis=(0,2),keepdims=True)
    data_normalized = (data - mean) / std

    BCICIV2a_Dataset = (data_normalized,labels)

    if pickle_dataset:
            with open(pickle_path,"wb") as f:
                pickle.dump(BCICIV2a_Dataset, f)

    return BCICIV2a_Dataset


def physionet_Dataset_Loader(path_to_dataset, subjects, tmin = 0, tmax = 4, resample_fs = 128,
                                fmin =1, fmax = 45, pick_6 = True,
                                CAR = True, filter = True, plot=False, plot_psd=False, do_faster = True,
                                pickle_dataset = False, pickle_path = None):
    """Creates a dataset for the subjects of the PhysioNet EEG Motor Movement/Imagery Dataset in the form of a 
    dictionary and in a within-subject approach where each subject's data and labels are seperate. 

    Args:
        path_to_dataset (str): Path to the folder where the raw files are stored which can be downloaded from https://physionet.org/content/eegmmidb/1.0.0/
        subjects (list): List containing integers in the range of 1 to 109 representing the subjects of the dataset to be processed
        tmin (float, optional): Epoching starts tmin before the cue onset. Defaults to 0.
        tmax (int, optional): Epoching ends tmax after the cue onset. Defaults to 4.
        resample_fs (int, optional): The frequency the signals are resampled to. Defaults to 128.
        fmin (int, optional): The lower cutoff frequency of the 5th order Butterworth bandpass filter. Defaults to 1.
        fmax (int, optional): The upper cutoff frequency of the 5th order Butterworth bandpass filter. Defaults to 45.
        pick_6 (bool, optional): Whether to keep only the 6 electrodes or to keep all 22 EEG electrodes. Defaults to True.
        CAR (bool, optional): Whether to apply common average referencing. Defaults to True.
        filter (bool, optional): Whether to apply bandpass filtering. Defaults to True.
        plot (bool, optional): Whether to generate a time series plot of the electrodes data after epoching. Defaults to False.
        plot_psd (bool, optional): Whether to plot the power spectral density before and after filterring. Defaults to False.
        do_faster (bool, optional): Whether to apply the FASTER algorithm. Defaults to True.
        pickle_dataset (bool, optional): Whether to generate a pickle object of the dataset so it can be loaded back again. Defaults to False.
        pickle_path (str, optional): Path to the destination where the pickle dataset object will be saved. Defaults to None.

    Returns:
        dict: The keys are integers corrosponding to the subject of the dataset. The values are tuples, where the first element is the epoched data
        of the subject in the shape (number of epochs, number of electrodes, number of time points), and the second element is the labels for each epoch in the shape (number of epochs,)
    """
    
    rename_mapping = {
        'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCz', 'Fc2.': 'FC2', 'Fc4.': 'FC4', 'Fc6.': 'FC6',
        'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1', 'Cz..': 'Cz', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6',
        'Cp5.': 'CP5', 'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPz', 'Cp2.': 'CP2', 'Cp4.': 'CP4', 'Cp6.': 'CP6',
        'Fp1.': 'Fp1', 'Fpz.': 'Fpz', 'Fp2.': 'Fp2', 'Af7.': 'AF7', 'Af3.': 'AF3', 'Afz.': 'AFz', 'Af4.': 'AF4', 'Af8.': 'AF8',
        'F7..': 'F7', 'F5..': 'F5', 'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'Fz', 'F2..': 'F2', 'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8',
        'Ft7.': 'FT7', 'Ft8.': 'FT8', 'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10', 'Tp7.': 'TP7', 'Tp8.': 'TP8',
        'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1', 'Pz..': 'Pz', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6', 'P8..': 'P8',
        'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POz', 'Po4.': 'PO4', 'Po8.': 'PO8', 'O1..': 'O1', 'Oz..': 'Oz', 'O2..': 'O2', 'Iz..': 'Iz'
    }
    montage = mne.channels.make_standard_montage('standard_1020')
    mne.set_log_level(verbose='CRITICAL')
    physio_dataset = {}
    
    for subject in Bar(f'Subjects',max=100).iter(subjects):
        if subject in (88,89,92,100,104,106):
            continue
        epochs = []
        for run in range(1,15):
            raw = mne.io.read_raw_edf(f'{path_to_dataset}//S{subject:03d}//s{subject:03d}R{run:02d}.edf',preload=True)
            raw.rename_channels(rename_mapping)
            raw.set_montage(montage)
            raw.info['bads'] = []
            eeg_picks = mne.pick_types(raw.info,eeg=True,eog=False)
            events = mne.events_from_annotations(raw)
            for key in events[1].keys():
                if key=='T1' and run in (4,8,12):
                    #left fist
                    lf_event = []
                    lf_dict = {'left_fist':1}
                    event = events[0].copy()
                    for i in range(len(event)):
                        if event[i][2] == events[1][key]:
                            event[i][2] = lf_dict['left_fist']
                            lf_event.append(event[i])
                    epoch = mne.Epochs(raw,lf_event,event_id=lf_dict,tmin = tmin,tmax = tmax,preload=True, baseline=None)
                    epoch.crop(tmin,tmax,include_tmax=False)
                    epochs.append(epoch)
                if key=='T1' and run in (6,10,14):
                    #both fists
                    bf_event = []
                    bf_dict = {'both_fists':4}
                    event = events[0].copy()
                    for i in range(len(event)):
                        if event[i][2] == events[1][key]:
                            event[i][2] = bf_dict['both_fists']
                            bf_event.append(event[i])
                    epoch = mne.Epochs(raw,bf_event,event_id=bf_dict,tmin = tmin,tmax = tmax,preload=True, baseline=None)
                    epoch.crop(tmin,tmax,include_tmax=False)
                    epochs.append(epoch)
                if key=='T2' and run in (4,8,12):
                    #right fist
                    rf_event = []
                    rf_dict = {'right_fist':2}
                    event = events[0].copy()
                    for i in range(len(event)):
                        if event[i][2] == events[1][key]:
                            event[i][2] = rf_dict['right_fist']
                            rf_event.append(event[i])
                    epoch = mne.Epochs(raw,rf_event,event_id=rf_dict,tmin = tmin,tmax = tmax,preload=True, baseline=None)
                    epoch.crop(tmin,tmax,include_tmax=False)
                    epochs.append(epoch)
                if key=='T2' and run in (6,10,14):
                    #both feet
                    bfe_event = []
                    bfe_dict = {'both_feet':3}
                    event = events[0].copy()
                    for i in range(len(event)):
                        if event[i][2] == events[1][key]:
                            event[i][2] = bfe_dict['both_feet']
                            bfe_event.append(event[i])
                    epoch = mne.Epochs(raw,bfe_event,event_id=bfe_dict,tmin = tmin,tmax = tmax,preload=True, baseline=None)
                    epoch.crop(tmin,tmax,include_tmax=False)
                    epochs.append(epoch)
                else:
                    pass
                pass
        epoch_total = mne.concatenate_epochs(epochs)

        if filter:  
            if plot_psd:
                unfilt_epoch_psd = epoch_total.compute_psd(remove_dc=True)
            epoch_total = epoch_total.filter(fmin,fmax,method='iir',iir_params=dict(order=5,ftype='butter'))
            if plot_psd:
                filt_epoch_psd = epoch_total.compute_psd(remove_dc=True)
        if plot_psd:
            fig, ax = plt.subplots()
            filt_epoch_psd.plot(color='r',spatial_colors=False,axes = ax)
            unfilt_epoch_psd.plot(spatial_colors=False,axes = ax)
            plt.show()
        
        if do_faster:
            epoch_total.info['bads'] = find_bad_channels(epoch_total,eeg_ref_corr=False)
            if len(epoch_total.info['bads']) > 0:
                epoch_total.interpolate_bads()
            
            bad_epochs = find_bad_epochs(epoch_total)
            if len(bad_epochs) > 0:
                epoch_total.drop(bad_epochs)

            
            ica = mne.preprocessing.ICA(0.99,method='infomax').fit(epoch_total, picks = eeg_picks)
            ica.exclude = find_bad_components(ica, epoch_total,use_metrics=['kurtosis', 'power_gradient',
                                                                            'hurst','median_gradient',
                                                                            'line_noise'])
            ica.apply(epoch_total)

            bad_channels_per_epoch = find_bad_channels_in_epochs(epoch_total, eeg_ref_corr=False)
            for i, b in enumerate(bad_channels_per_epoch):
                if len(b) > 0:
                    ep = epoch_total[i]
                    ep.info['bads'] = b
                    ep.interpolate_bads() 
                    epoch_total._data[i, :, :] = ep._data[0, :, :]
        
        if CAR:
            epoch_total.set_eeg_reference('average')
        
        epoch_total.resample(resample_fs)
        if pick_6:
            epoch_total.pick(['C5','C3','C1','C2','C4','C6'])
        epoch_total.equalize_event_counts()

        if plot:
            epoch_total.plot(n_channels = 6,n_epochs=5,events=True,block=True,event_id=True,scalings=dict(eeg = 90e-2))
        
        labels = epoch_total.events[:,-1]
        data = epoch_total.get_data()
        mean = np.mean(data,axis=(0,2),keepdims=True)
        std = np.std(data,axis=(0,2),keepdims=True)
        data_normalized = (data - mean) / std
        physio_dataset[subject] = ([data_normalized,labels])

    if pickle_dataset:
        with open(pickle_path,"wb") as f:
            pickle.dump(physio_dataset, f)
    return physio_dataset

def physionet_Transfer_Dataset_Loader(path_to_dataset, subjects, tmin = 0, tmax = 4, resample_fs = 128,
                                fmin =1, fmax = 45, pick_6 = True,
                                CAR = True, filter = True, plot=False, plot_psd=False, do_faster = True,
                                pickle_dataset = False, pickle_path = None):
    """Creates a concatenated dataset for the subjects of the PhysioNet EEG Motor Movement/Imagery Dataset in the form of a 
    tuple and in a cross-subject. 

    Args:
        path_to_dataset (str): Path to the folder where the raw files are stored which can be downloaded from https://physionet.org/content/eegmmidb/1.0.0/
        subjects (list): List containing integers in the range of 1 to 109 representing the subjects of the dataset to be processed
        tmin (float, optional): Epoching starts tmin before the cue onset. Defaults to 0.
        tmax (int, optional): Epoching ends tmax after the cue onset. Defaults to 4.
        resample_fs (int, optional): The frequency the signals are resampled to. Defaults to 128.
        fmin (int, optional): The lower cutoff frequency of the 5th order Butterworth bandpass filter. Defaults to 1.
        fmax (int, optional): The upper cutoff frequency of the 5th order Butterworth bandpass filter. Defaults to 45.
        pick_6 (bool, optional): Whether to keep only the 6 electrodes or to keep all 22 EEG electrodes. Defaults to True.
        CAR (bool, optional): Whether to apply common average referencing. Defaults to True.
        filter (bool, optional): Whether to apply bandpass filtering. Defaults to True.
        plot (bool, optional): Whether to generate a time series plot of the electrodes data after epoching. Defaults to False.
        plot_psd (bool, optional): Whether to plot the power spectral density before and after filterring. Defaults to False.
        do_faster (bool, optional): Whether to apply the FASTER algorithm. Defaults to True.
        pickle_dataset (bool, optional): Whether to generate a pickle object of the dataset so it can be loaded back again. Defaults to False.
        pickle_path (str, optional): Path to the destination where the pickle dataset object will be saved. Defaults to None.

    Returns:
        tuple: The first element is the epoched data
        of the concatenated subjects in the shape (number of epochs, number of electrodes, number of time points), and the second element is the labels for each epoch in the shape (number of epochs,)
    """
    rename_mapping = {
        'Fc5.': 'FC5', 'Fc3.': 'FC3', 'Fc1.': 'FC1', 'Fcz.': 'FCz', 'Fc2.': 'FC2', 'Fc4.': 'FC4', 'Fc6.': 'FC6',
        'C5..': 'C5', 'C3..': 'C3', 'C1..': 'C1', 'Cz..': 'Cz', 'C2..': 'C2', 'C4..': 'C4', 'C6..': 'C6',
        'Cp5.': 'CP5', 'Cp3.': 'CP3', 'Cp1.': 'CP1', 'Cpz.': 'CPz', 'Cp2.': 'CP2', 'Cp4.': 'CP4', 'Cp6.': 'CP6',
        'Fp1.': 'Fp1', 'Fpz.': 'Fpz', 'Fp2.': 'Fp2', 'Af7.': 'AF7', 'Af3.': 'AF3', 'Afz.': 'AFz', 'Af4.': 'AF4', 'Af8.': 'AF8',
        'F7..': 'F7', 'F5..': 'F5', 'F3..': 'F3', 'F1..': 'F1', 'Fz..': 'Fz', 'F2..': 'F2', 'F4..': 'F4', 'F6..': 'F6', 'F8..': 'F8',
        'Ft7.': 'FT7', 'Ft8.': 'FT8', 'T7..': 'T7', 'T8..': 'T8', 'T9..': 'T9', 'T10.': 'T10', 'Tp7.': 'TP7', 'Tp8.': 'TP8',
        'P7..': 'P7', 'P5..': 'P5', 'P3..': 'P3', 'P1..': 'P1', 'Pz..': 'Pz', 'P2..': 'P2', 'P4..': 'P4', 'P6..': 'P6', 'P8..': 'P8',
        'Po7.': 'PO7', 'Po3.': 'PO3', 'Poz.': 'POz', 'Po4.': 'PO4', 'Po8.': 'PO8', 'O1..': 'O1', 'Oz..': 'Oz', 'O2..': 'O2', 'Iz..': 'Iz'
    }
    montage = mne.channels.make_standard_montage('standard_1020')
    mne.set_log_level(verbose='CRITICAL')
    transfer_learning_epochs = []

    for subject in Bar(f'Subjects',max=100).iter(subjects):
        if subject in (88,89,92,100,104,106):
            continue
        epochs = []
        for run in range(1,15):
            raw = mne.io.read_raw_edf(f'{path_to_dataset}//S{subject:03d}//s{subject:03d}R{run:02d}.edf',preload=True)
            raw.rename_channels(rename_mapping)
            raw.set_montage(montage)
            raw.info['bads'] = []
            eeg_picks = mne.pick_types(raw.info,eeg=True,eog=False)
            events = mne.events_from_annotations(raw)
            for key in events[1].keys():
                if key=='T1' and run in (4,8,12):
                    #left fist
                    lf_event = []
                    lf_dict = {'left_fist':1}
                    event = events[0].copy()
                    for i in range(len(event)):
                        if event[i][2] == events[1][key]:
                            event[i][2] = lf_dict['left_fist']
                            lf_event.append(event[i])
                    epoch = mne.Epochs(raw,lf_event,event_id=lf_dict,tmin = tmin,tmax = tmax,preload=True, baseline=None)
                    epoch.crop(tmin,tmax,include_tmax=False)
                    epochs.append(epoch)
                if key=='T1' and run in (6,10,14):
                    #both fists
                    bf_event = []
                    bf_dict = {'both_fists':4}
                    event = events[0].copy()
                    for i in range(len(event)):
                        if event[i][2] == events[1][key]:
                            event[i][2] = bf_dict['both_fists']
                            bf_event.append(event[i])
                    epoch = mne.Epochs(raw,bf_event,event_id=bf_dict,tmin = tmin,tmax = tmax,preload=True, baseline=None)
                    epoch.crop(tmin,tmax,include_tmax=False)
                    epochs.append(epoch)
                if key=='T2' and run in (4,8,12):
                    #right fist
                    rf_event = []
                    rf_dict = {'right_fist':2}
                    event = events[0].copy()
                    for i in range(len(event)):
                        if event[i][2] == events[1][key]:
                            event[i][2] = rf_dict['right_fist']
                            rf_event.append(event[i])
                    epoch = mne.Epochs(raw,rf_event,event_id=rf_dict,tmin = tmin,tmax = tmax,preload=True, baseline=None)
                    epoch.crop(tmin,tmax,include_tmax=False)
                    epochs.append(epoch)
                if key=='T2' and run in (6,10,14):
                    #both feet
                    bfe_event = []
                    bfe_dict = {'both_feet':3}
                    event = events[0].copy()
                    for i in range(len(event)):
                        if event[i][2] == events[1][key]:
                            event[i][2] = bfe_dict['both_feet']
                            bfe_event.append(event[i])
                    epoch = mne.Epochs(raw,bfe_event,event_id=bfe_dict,tmin = tmin,tmax = tmax,preload=True, baseline=None)
                    epoch.crop(tmin,tmax,include_tmax=False)
                    epochs.append(epoch)
                else:
                    pass
        epoch_total = mne.concatenate_epochs(epochs)
        
        if filter:  
            if plot_psd:
                unfilt_epoch_psd = epoch_total.compute_psd(remove_dc=True)
            epoch_total = epoch_total.filter(fmin,fmax,method='iir',iir_params=dict(order=5,ftype='butter'))
            if plot_psd:
                filt_epoch_psd = epoch_total.compute_psd(remove_dc=True)
        if plot_psd:
            fig, ax = plt.subplots()
            filt_epoch_psd.plot(color='r',spatial_colors=False,axes = ax)
            unfilt_epoch_psd.plot(spatial_colors=False,axes = ax)
            plt.show()
        
        if do_faster:
            epoch_total.info['bads'] = find_bad_channels(epoch_total,eeg_ref_corr=False)
            if len(epoch_total.info['bads']) > 0:
                epoch_total.interpolate_bads()
            
            bad_epochs = find_bad_epochs(epoch_total)
            if len(bad_epochs) > 0:
                epoch_total.drop(bad_epochs)

            
            ica = mne.preprocessing.ICA(0.99,method='infomax').fit(epoch_total, picks = eeg_picks)
            ica.exclude = find_bad_components(ica, epoch_total,use_metrics=['kurtosis', 'power_gradient',
                                                                            'hurst','median_gradient',
                                                                            'line_noise'])
            ica.apply(epoch_total)

            bad_channels_per_epoch = find_bad_channels_in_epochs(epoch_total, eeg_ref_corr=False)
            for i, b in enumerate(bad_channels_per_epoch):
                if len(b) > 0:
                    ep = epoch_total[i]
                    ep.info['bads'] = b
                    ep.interpolate_bads() 
                    epoch_total._data[i, :, :] = ep._data[0, :, :]
        
        if CAR:
            epoch_total.set_eeg_reference('average')

        epoch_total.resample(resample_fs)
        if pick_6:
            epoch_total.pick(['C5','C3','C1','C2','C4','C6'])
        
        if plot:
            epoch_total.plot(n_channels = 6,n_epochs=5,events=True,block=True,event_id=True,scalings=dict(eeg = 20e-6))

        transfer_learning_epochs.append(epoch_total)

    total_transfer_learning_epochs = mne.concatenate_epochs(transfer_learning_epochs) 
    total_transfer_learning_epochs.equalize_event_counts()   

    labels = total_transfer_learning_epochs.events[:,-1]
    data = total_transfer_learning_epochs.get_data()
    mean = np.mean(data,axis=(0,2),keepdims=True)
    std = np.std(data,axis=(0,2),keepdims=True)
    data_normalized = (data - mean) / std
    physio_dataset = (data_normalized,labels)

    if pickle_dataset:
            with open(pickle_path,"wb") as f:
                pickle.dump(physio_dataset, f)

    return physio_dataset


def Mindrove_Dataset_Loader(path_to_dataset, subjects,tmin = 0,tmax = 5,fmin =1, fmax = 45, resample_fs = 128,
                                mark_bads = False, load_bad_indices = False,
                                CAR = True, filter = True, plot=False, plot_psd=False,
                                pickle_dataset = False, pickle_path = None):
    """Creates a dataset for the subjects of the MindRove dataset in the form of a 
    dictionary and in a within-subject approach where each subject's data and labels are seperate. 

    Args:
        path_to_dataset (str): Path to the folder where the raw files are stored which can be downloaded from https://github.com/Ramiz03S/MindRove_Dataset
        subjects (list): List containing integers in the range of 1 to 6 representing the subjects of the dataset to be processed
        tmin (float, optional): Epoching starts tmin before the cue onset. Defaults to 0.
        tmax (int, optional): Epoching ends tmax after the cue onset. Defaults to 5.
        fmin (int, optional): The lower cutoff frequency of the 5th order Butterworth bandpass filter. Defaults to 1.
        fmax (int, optional): The upper cutoff frequency of the 5th order Butterworth bandpass filter. Defaults to 45.
        resample_fs (int, optional): The frequency the signals are resampled to. Defaults to 128.
        mark_bads (bool, optional): Whether to save the epochs marked red while plotting to a JSON file. Defaults to False.
        load_bad_indices (bool, optional): Whether to load the epochs marked bad from the JSON file and drop them from the dataset. Defaults to False.
        CAR (bool, optional): Whether to apply common average referencing. Defaults to True.
        filter (bool, optional): Whether to apply bandpass filtering. Defaults to True.
        plot (bool, optional): Whether to generate a time series plot of the electrodes data after epoching. Defaults to False.
        plot_psd (bool, optional): Whether to plot the power spectral density before and after filterring. Defaults to False.
        do_faster (bool, optional): Whether to apply the FASTER algorithm. Defaults to True.
        pickle_dataset (bool, optional): Whether to generate a pickle object of the dataset so it can be loaded back again. Defaults to False.
        pickle_path (str, optional): Path to the destination where the pickle dataset object will be saved. Defaults to None.

    Returns:
        dict: The keys are integers corrosponding to the subject of the dataset. The values are tuples, where the first element is the epoched data
        of the subject in the shape (number of epochs, number of electrodes, number of time points), and the second element is the labels for each epoch in the shape (number of epochs,)
    """
    mne.set_log_level(verbose='CRITICAL')
    MindRove_Dataset = {}
    fs = 500
    n_channels = 6
    classes = ["right_left","tongue_feet"]
    event_id={'left fist':1,'right fist':2,'both feet':3, 'tongue':4}
    channel_names = [f'C{i+1}' for i in range(n_channels)]
    channel_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=channel_names, ch_types=channel_types,sfreq=fs)
    bad_epochs_subject_dict = defaultdict(list)
    if load_bad_indices:
        with open('MindRove_bad_epoch_indices.json', 'r') as f:
            loaded_bad_epochs_subject_dict = json.load(f)
        bad_epochs_subject_dict.update(loaded_bad_epochs_subject_dict)

    for subject in Bar(f'subjects',max=6).iter(subjects):
        epochs_concat = []
        for class_type in classes:
            for run in range(1,4):
                raw_csv = np.loadtxt(fname=f'{path_to_dataset}//S_{subject:02d}//{class_type}//run_{run}.csv', delimiter=',')
                raw = mne.io.RawArray(data=((raw_csv[:,:6]).T)*(4.5e-8),info=info)
                events_times = raw_csv[:,-1]
                events = []
                for i, x in enumerate(events_times):
                    if class_type == "right_left":
                        if x == 1:
                            events.append([i,0,2]) #right
                        if x == 2:
                            events.append([i,0,1]) #left
                    if class_type == "tongue_feet":
                        if x == 1:
                            events.append([i,0,4]) #tongue
                        if x == 2:
                            events.append([i,0,3]) #feet
                epochs = mne.Epochs(raw,events,event_id=event_id,
                                tmin = tmin,tmax = tmax,preload=True,
                                baseline=None, on_missing='ignore')
                epochs = epochs.crop(tmin=tmin,tmax=tmax,include_tmax= False)
                epochs_concat.append(epochs)

        epochs_total = mne.concatenate_epochs(epochs_concat)

        if filter:  
            if plot_psd:
                unfilt_epoch_psd = epochs_total.compute_psd(remove_dc=True)
            epochs_total = epochs_total.filter(fmin,fmax,method='iir',iir_params=dict(order=5,ftype='butter'))
            if plot_psd:
                filt_epoch_psd = epochs_total.compute_psd(remove_dc=True)      
        if plot_psd:
            fig, ax = plt.subplots()
            filt_epoch_psd.plot(color='r',spatial_colors=False,axes = ax)
            unfilt_epoch_psd.plot(spatial_colors=False,axes = ax)
            plt.show()

        epochs_total.resample(resample_fs)

        if CAR:
            epochs_total.set_eeg_reference('average')
        
        if plot:
            epochs_total.plot(n_channels = 6,n_epochs=5,events=epochs_total.events,block=True,
                        event_id=event_id) 
        if mark_bads:
            bad_epoch_indicies = [i for i, x in enumerate(epochs_total.drop_log) if x]
            bad_epochs_subject_dict[f'{subject}'] = bad_epoch_indicies
            with open('MindRove_bad_epoch_indices.json', 'w') as f:
                json.dump(bad_epochs_subject_dict, f)
        
        if load_bad_indices:
            epochs_total.drop(bad_epochs_subject_dict[f'{subject}'])
        epochs_total.equalize_event_counts()
        epochs_total.reorder_channels(['C5','C3','C1','C2','C4','C6'])
        labels = epochs_total.events[:,-1]
        data = epochs_total.get_data()

        mean = np.mean(data,axis=(0,2),keepdims=True)
        std = np.std(data,axis=(0,2),keepdims=True)
        data_normalized = (data - mean) / std
        MindRove_Dataset[subject] = ([data_normalized,labels])
    
    if pickle_dataset:
        with open(pickle_path,"wb") as f:
            pickle.dump(MindRove_Dataset, f)
    return MindRove_Dataset   

def Mindrove_Transfer_Dataset_Loader(path_to_dataset, subjects,tmin = 0,tmax = 5,fmin =1, fmax = 45,resample_fs = 128,
                                load_bad_indices = False,
                                CAR = True, filter = True, plot=False, plot_psd=False,
                                pickle_dataset = False, pickle_path = None):
    """Creates a concatenated dataset for the subjects of the MindRove Dataset in the form of a 
    tuple and in a cross-subject. 

    Args:
        path_to_dataset (str): Path to the folder where the raw files are stored which can be downloaded from https://github.com/Ramiz03S/MindRove_Dataset
        subjects (list): List containing integers in the range of 1 to 6 representing the subjects of the dataset to be processed
        tmin (float, optional): Epoching starts tmin before the cue onset. Defaults to 0.
        tmax (int, optional): Epoching ends tmax after the cue onset. Defaults to 5.
        fmin (int, optional): The lower cutoff frequency of the 5th order Butterworth bandpass filter. Defaults to 1.
        fmax (int, optional): The upper cutoff frequency of the 5th order Butterworth bandpass filter. Defaults to 45.
        resample_fs (int, optional): The frequency the signals are resampled to. Defaults to 128.
        mark_bads (bool, optional): Whether to save the epochs marked red while plotting to a JSON file. Defaults to False.
        load_bad_indices (bool, optional): Whether to load the epochs marked bad from the JSON file and drop them from the dataset. Defaults to False.
        CAR (bool, optional): Whether to apply common average referencing. Defaults to True.
        filter (bool, optional): Whether to apply bandpass filtering. Defaults to True.
        plot (bool, optional): Whether to generate a time series plot of the electrodes data after epoching. Defaults to False.
        plot_psd (bool, optional): Whether to plot the power spectral density before and after filterring. Defaults to False.
        do_faster (bool, optional): Whether to apply the FASTER algorithm. Defaults to True.
        pickle_dataset (bool, optional): Whether to generate a pickle object of the dataset so it can be loaded back again. Defaults to False.
        pickle_path (str, optional): Path to the destination where the pickle dataset object will be saved. Defaults to None.

    Returns:
        tuple: The first element is the epoched data
        of the concatenated subjects in the shape (number of epochs, number of electrodes, number of time points), and the second element is the labels for each epoch in the shape (number of epochs,)
    """
    mne.set_log_level(verbose='CRITICAL')
    fs = 500
    n_channels = 6
    classes = ["right_left","tongue_feet"]
    event_id={'left fist':1,'right fist':2,'both feet':3, 'tongue':4}
    channel_names = [f'C{i+1}' for i in range(n_channels)]
    channel_types = ['eeg'] * n_channels
    info = mne.create_info(ch_names=channel_names, ch_types=channel_types,sfreq=fs)
    bad_epochs_subject_dict = defaultdict(list)
    if load_bad_indices:
        with open('MindRove_bad_epoch_indices.json', 'r') as f:
            loaded_bad_epochs_subject_dict = json.load(f)
        bad_epochs_subject_dict.update(loaded_bad_epochs_subject_dict)

    epoch_subjects = []
    for subject in Bar(f'subjects',max=6).iter(subjects):
        epochs_concat = []
        for class_type in classes:
            for run in range(1,4):
                raw_csv = np.loadtxt(fname=f'{path_to_dataset}//S_{subject:02d}//{class_type}//run_{run}.csv', delimiter=',')
                raw = mne.io.RawArray(data=((raw_csv[:,:6]).T)*(4.5e-8),info=info)
                events_times = raw_csv[:,-1]
                events = []
                for i, x in enumerate(events_times):
                    if class_type == "right_left":
                        if x == 1:
                            events.append([i,0,2]) #right
                        if x == 2:
                            events.append([i,0,1]) #left
                    if class_type == "tongue_feet":
                        if x == 1:
                            events.append([i,0,4]) #tongue
                        if x == 2:
                            events.append([i,0,3]) #feet
                epochs = mne.Epochs(raw,events,event_id=event_id,
                                tmin = tmin,tmax = tmax,preload=True,
                                baseline=None, on_missing='ignore')
                epochs = epochs.crop(tmin=tmin,tmax=tmax,include_tmax= False)
                epochs_concat.append(epochs)
        epoch_subject = mne.concatenate_epochs(epochs_concat)
        
        epoch_subject.drop(bad_epochs_subject_dict[f'{subject}'])
        epoch_subjects.append(epoch_subject)

    epochs_total = mne.concatenate_epochs(epoch_subjects)

    if filter:  
        if plot_psd:
            unfilt_epoch_psd = epochs_total.compute_psd(remove_dc=True)
        epochs_total = epochs_total.filter(fmin,fmax,method='iir',iir_params=dict(order=5,ftype='butter'))
        if plot_psd:
            filt_epoch_psd = epochs_total.compute_psd(remove_dc=True)      
    if plot_psd:
        fig, ax = plt.subplots()
        filt_epoch_psd.plot(color='r',spatial_colors=False,axes = ax)
        unfilt_epoch_psd.plot(spatial_colors=False,axes = ax)
        plt.show()

    epochs_total.resample(resample_fs)

    if CAR:
        epochs_total.set_eeg_reference('average')
    
    if plot:
        epochs_total.plot(n_channels = 6,n_epochs=5,events=epochs_total.events,block=True,
                    event_id=event_id
                    ) #,scalings=dict(eeg = 20e-6)

    epochs_total.reorder_channels(['C5','C3','C1','C2','C4','C6'])
    labels = epochs_total.events[:,-1]
    data = epochs_total.get_data()

    mean = np.mean(data,axis=(0,2),keepdims=True)
    std = np.std(data,axis=(0,2),keepdims=True)
    data_normalized = (data - mean) / std

    MindRove_Dataset = (data_normalized,labels)
    
    if pickle_dataset:
        with open(pickle_path,"wb") as f:
            pickle.dump(MindRove_Dataset, f)
    return MindRove_Dataset

