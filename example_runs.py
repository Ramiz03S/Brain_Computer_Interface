from Dataset_Loader import BCICIV2a_Dataset_Loader, BCICIV2a_Transfer_Dataset_Loader, Mindrove_Dataset_Loader, Mindrove_Transfer_Dataset_Loader, physionet_Dataset_Loader, physionet_Transfer_Dataset_Loader
from within_subject_training import run_within_subject
from transfer_learning_training import run_transfer_learning

if __name__ == '__main__':
    BCICIV2a_Dataset_Loader('BCICIV2a',[1,2,3,4,5,6,7,8,9],fmin=5,fmax=30,
                            pickle_dataset=True,pickle_path='BCICdataset.pkl')
    
    physionet_Dataset_Loader('Physionet',[1,15,25,35,43,55,65,73,85],fmin=5,fmax=30,
                            pickle_dataset=True,pickle_path='PhysioNetdataset.pkl')

    Mindrove_Dataset_Loader('MindRove',[1,2,3,4,5,6],fmin=5,fmax=30,pickle_dataset=True,
                            pickle_path='MindRovedataset.pkl',load_bad_indices=True)
    
    run_within_subject(model_type='EEGNet',dataset_type='BCICIV2a',pickle_path='BCICdataset.pkl',
                       lr=0.001,EPOCHS=500,PATIENCE=100,GIVE_UP=100,MIN_DELTA=0.05,
                       window_series_ratio=1,window_stride_ratio=1)
    
    run_within_subject(model_type='EEGNet',dataset_type='Physionet',pickle_path='PhysioNetdataset.pkl',
                       lr=0.001,EPOCHS=500,PATIENCE=100,GIVE_UP=100,MIN_DELTA=0.05,
                       window_series_ratio=1,window_stride_ratio=1)
    
    run_within_subject(model_type='EEGNet',dataset_type='MindRove',pickle_path='MindRovedataset.pkl',
                       lr=0.001,EPOCHS=500,PATIENCE=100,GIVE_UP=100,MIN_DELTA=0.05,
                       window_series_ratio=1,window_stride_ratio=1)

    run_transfer_learning(model_type='EEGNet',dataset_type='BCICIV2a',lr=0.001,EPOCHS=500,
                          PATIENCE=100,GIVE_UP=100,MIN_DELTA=0.05,window_series_ratio=1,window_stride_ratio=1)
    
    run_transfer_learning(model_type='EEGNet',dataset_type='physionet',lr=0.001,EPOCHS=500,
                          PATIENCE=100,GIVE_UP=100,MIN_DELTA=0.05,window_series_ratio=1,window_stride_ratio=1)
    
    run_transfer_learning(model_type='EEGNet',dataset_type='MindRove',lr=0.001,EPOCHS=500,
                          PATIENCE=100,GIVE_UP=100,MIN_DELTA=0.05,window_series_ratio=1,window_stride_ratio=1)

    