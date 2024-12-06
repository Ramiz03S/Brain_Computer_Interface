import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from Dataset_Loader import physionet_Dataset_Loader
from Dataset_Loader import BCICIV2a_Dataset_Loader
from Dataset_Loader import Mindrove_Dataset_Loader
import numpy as np
import pickle
from misc import window_augment, give_up, within_subject_dataset, model_config, metric_reporting, metric_calculation

def within_subject_training(dataset, subject, lr, EPOCHS,PATIENCE,GIVE_UP, model_type,dataset_type,
                            parameter_hypertuning=True,full_training=True,
                            TEST_SIZE = 0.2, BATCH_SIZE = 32,MIN_DELTA = 0.01,
                            window_series_ratio = 0.25, window_stride_ratio = 1):
    
    X, y = (dataset[subject][0],dataset[subject][1])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE,shuffle=True, random_state=0,stratify=y)
    
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)

    if parameter_hypertuning:
        fold_metrics = {'accuracy':[],'precision_score':[],'recall_score':[],
                        'f1_score':[],'confusion_matrix':[],'cohen_kappa_score':[],
                        'stopped_epoch':[], 'validation_loss':[], 'training_loss': []}
        validation_metrics = fold_metrics.copy()

        for fold, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
            os.makedirs(f'checkpoint_{dataset_type}//{model_type}//within_subject//subject_{subject}//val_fold_{fold}', exist_ok=True)
            
            X_train_fold, y_train_fold = X_train[train_index], y_train[train_index]
            X_val_fold, y_val_fold = X_train[val_index], y_train[val_index]
            (train_fold,val_fold) = window_augment(window_series_ratio,window_stride_ratio,(X_train_fold,y_train_fold),(X_val_fold, y_val_fold))
            
            time_points_length = np.shape(train_fold[0])[2]
            number_channels = np.shape(train_fold[0])[1]
            
            train_dataset = within_subject_dataset(train_fold)
            val_dataset = within_subject_dataset(val_fold)

            train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=False)
            val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)

            checkpoint = keras.callbacks.ModelCheckpoint(
                f'checkpoint_{dataset_type}//{model_type}//within_subject//subject_{subject}//val_fold_{fold}//model.keras',
                monitor='val_loss',
                verbose=1,
                mode='min',
                save_best_only=True)
            earlystopping = give_up(give_up=GIVE_UP,patience=PATIENCE,min_delta=MIN_DELTA)
            logger = keras.callbacks.CSVLogger(f'checkpoint_{dataset_type}//{model_type}//within_subject//subject_{subject}//val_fold_{fold}//training.log',append=False)
            callbackslist = [checkpoint, earlystopping, logger]
            
            optimizer = keras.optimizers.Adam(learning_rate=lr)
            lossfn = keras.losses.categorical_crossentropy
            model = model_config(number_channels,model_type,time_points_length,0.5)
            
            model.compile(loss=lossfn,optimizer=optimizer,metrics=['accuracy'])
            history = model.fit(train_loader,epochs=EPOCHS,validation_data=val_loader,callbacks=callbackslist)
            
            best_model = keras.saving.load_model(f'checkpoint_{dataset_type}//{model_type}//within_subject//subject_{subject}//val_fold_{fold}//model.keras',compile=True)

            pred_prob_y = best_model.predict(val_loader)
            y_pred = np.argmax(pred_prob_y, axis=1)
            y_true = val_dataset.y_true()
            stopped_epoch = earlystopping.stopped_epoch + 1
            fold_metrics = metric_calculation(fold_metrics, y_true, y_pred)
            fold_metrics['stopped_epoch'].append(stopped_epoch)
            fold_metrics['validation_loss'].append(history.history['val_loss'])
            fold_metrics['training_loss'].append(history.history['loss'])

        for i in fold_metrics.keys():
            if i == 'confusion_matrix':
                validation_metrics[i] = np.sum(fold_metrics[i], axis=0)
            elif i == 'stopped_epoch':
                validation_metrics[i] = int(np.mean(fold_metrics[i], axis=0))
            elif i == 'training_loss' or i == 'validation_loss':
                validation_metrics[i] = fold_metrics[i]
            else:
                validation_metrics[i] = (np.mean(fold_metrics[i]),np.std(fold_metrics[i]))
        validation_metrics['subject'] = subject

        with open(f'checkpoint_{dataset_type}//{model_type}//within_subject//subject_{subject}//validation_metrics.pkl',"wb") as f:
                pickle.dump(validation_metrics, f)
    
    if full_training:
        with open(f'checkpoint_{dataset_type}//{model_type}//within_subject//subject_{subject}//validation_metrics.pkl',"rb") as f:
                validation_metrics = pickle.load(f)
        test_metrics = {'accuracy':[],'precision_score':[],'recall_score':[],
                        'f1_score':[],'confusion_matrix':[],'cohen_kappa_score':[],
                        'stopped_epoch':[]}

        entire_train = window_augment(window_series_ratio,window_stride_ratio,(X_train,y_train))
        entire_train_dataset = within_subject_dataset(entire_train[0])
        entire_train_loader = torch.utils.data.DataLoader(entire_train_dataset,batch_size=BATCH_SIZE,shuffle=True)

        time_points_length = np.shape(entire_train[0][0])[2]
        number_channels = np.shape(entire_train[0][0])[1]

        test = window_augment(window_series_ratio,window_stride_ratio,(X_test,y_test))
        test_dataset = within_subject_dataset(test[0])
        test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

        checkpoint = keras.callbacks.ModelCheckpoint(
            f'checkpoint_{dataset_type}//{model_type}//within_subject//subject_{subject}//full_model.keras',
            monitor='loss',
            verbose=1,
            mode='min',
            save_best_only=True)
        logger = keras.callbacks.CSVLogger(f'checkpoint_{dataset_type}//{model_type}//within_subject//subject_{subject}//full_model_training.log',append=False)
        callbackslist = [checkpoint, logger]
        
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        lossfn = keras.losses.categorical_crossentropy
        model = model_config(number_channels,model_type,time_points_length,0.5)
        model.compile(loss=lossfn,optimizer=optimizer,metrics=['accuracy'])
        model.fit(entire_train_loader,epochs=(validation_metrics['stopped_epoch']),callbacks=callbackslist)
        
        best_model = keras.saving.load_model(f'checkpoint_{dataset_type}//{model_type}//within_subject//subject_{subject}//full_model.keras',compile=True)
        pred_prob_y = best_model.predict(test_loader)
        y_pred = np.argmax(pred_prob_y, axis=1)
        y_true = test_dataset.y_true()

        test_metrics = metric_calculation(test_metrics,y_true,y_pred)
        test_metrics['subject'] = subject

        with open(f'checkpoint_{dataset_type}//{model_type}//within_subject//subject_{subject}//test_metrics.pkl',"wb") as f:
                pickle.dump(test_metrics, f)

def run_within_subject(model_type,dataset_type, pickle_path, lr, EPOCHS,PATIENCE,GIVE_UP,MIN_DELTA,window_series_ratio, window_stride_ratio):
    """runs within-subject training on the subjects of the dataset

    Args:
        model_type (str): Name of the nerual network model to use. Only 'EEGNet' implemented.
        dataset_type (str): Name of the dataset. It will be used to name the folders generated
        pickle_path (str): Path leading to the .pkl object where the dataset is stored
        lr (float): The learning rate of the neural network
        EPOCHS (int): The maximum number of training iterations
        PATIENCE (int): The maximum number of training iterations without a MIN_DELTA decrease in validation loss
        GIVE_UP (int): The maximum number of training iterations without the validation loss improving from the initial value
        MIN_DELTA (float): The decrease in validation loss that will reset the PATIENCE parameter
        window_series_ratio (float): The ratio between the sliding window's window length and the length of the time points of the epochs
        window_stride_ratio (float): The ratio between the sliding window's length and stride
    """
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    for subject in dataset.keys():
        within_subject_training(dataset, subject, lr, EPOCHS,PATIENCE,GIVE_UP, model_type,dataset_type,
                            parameter_hypertuning=True,full_training=True,
                            TEST_SIZE = 0.2, BATCH_SIZE = 32,MIN_DELTA=MIN_DELTA,
                            window_series_ratio=window_series_ratio, window_stride_ratio=window_stride_ratio)
    
        with open(f'checkpoint_{dataset_type}//{model_type}//within_subject//subject_{subject}//validation_metrics.pkl', 'rb') as f:
            validation_metrics = pickle.load(f)
            metric_reporting(validation_metrics,f'checkpoint_{dataset_type}//{model_type}//within_subject//metric_reporting.log',
                             dataset_type,model_type,'within_subject',
                              is_validation= True, do_loss_vs_epoch=True)
        with open(f'checkpoint_{dataset_type}//{model_type}//within_subject//subject_{subject}//test_metrics.pkl', 'rb') as f:
            test_metrics = pickle.load(f)
            metric_reporting(test_metrics,f'checkpoint_{dataset_type}//{model_type}//within_subject//metric_reporting.log',
                             dataset_type,model_type,'within_subject',
                              is_validation= False, do_loss_vs_epoch=False)
