import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from Dataset_Loader import physionet_Dataset_Loader, physionet_Transfer_Dataset_Loader
from Dataset_Loader import BCICIV2a_Dataset_Loader, BCICIV2a_Transfer_Dataset_Loader
from Dataset_Loader import Mindrove_Dataset_Loader, Mindrove_Transfer_Dataset_Loader
import numpy as np
import pickle
from misc import window_augment, give_up, within_subject_dataset, model_config, metric_reporting, metric_calculation

def transfer_learning_training(train_dataset,holdout_subject, lr, EPOCHS,PATIENCE,GIVE_UP, model_type,dataset_type,
                            hyperparameter_tuning=True,full_training=True,
                            BATCH_SIZE = 32,MIN_DELTA = 0.01,
                            window_series_ratio = 0.25, window_stride_ratio = 1):
    
    
    X, y = (train_dataset[0], train_dataset[1])
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)

    if hyperparameter_tuning:
        fold_metrics = {'accuracy':[],'precision_score':[],'recall_score':[],
                    'f1_score':[],'confusion_matrix':[],'cohen_kappa_score':[],
                    'stopped_epoch':[], 'validation_loss':[], 'training_loss': []}
        validation_metrics = fold_metrics.copy()

        for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
            os.makedirs(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//val_fold_{fold}', exist_ok=True)
        
            X_train_fold, y_train_fold = X[train_index], y[train_index]
            X_val_fold, y_val_fold = X[val_index], y[val_index]
            (train_fold,val_fold) = window_augment(window_series_ratio,window_stride_ratio,(X_train_fold,y_train_fold),(X_val_fold, y_val_fold))
            
            time_points_length = np.shape(train_fold[0])[2]
            number_channels = np.shape(train_fold[0])[1]
            
            train_dataset = within_subject_dataset(train_fold)
            val_dataset = within_subject_dataset(val_fold)

            train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=False)
            val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)
            
            checkpoint = keras.callbacks.ModelCheckpoint(
                f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//val_fold_{fold}//model_transfer_learning.keras',
                monitor='val_loss',
                verbose=1,
                mode='min',
                save_best_only=True,)
            earlystopping = give_up(give_up=GIVE_UP,patience=PATIENCE,min_delta=MIN_DELTA)
            logger = keras.callbacks.CSVLogger(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//val_fold_{fold}//transfer_training.log',append=False)
            callbackslist = [checkpoint, earlystopping, logger]
            
            optimizer = keras.optimizers.Adam(learning_rate=lr)
            lossfn = keras.losses.categorical_crossentropy
            model = model_config(number_channels,model_type,time_points_length,0.25)

            model.compile(loss=lossfn,optimizer=optimizer,metrics=['accuracy'])
            history = model.fit(train_loader,epochs=EPOCHS,validation_data=val_loader,callbacks=callbackslist)

            best_model = keras.saving.load_model(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//val_fold_{fold}//model_transfer_learning.keras',compile=True)

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

        with open(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//validation_metrics.pkl',"wb") as f:
                    pickle.dump(validation_metrics, f)

    if full_training:
        with open(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//validation_metrics.pkl',"rb") as f:
            validation_metrics = pickle.load(f)

        entire_train = window_augment(window_series_ratio,window_stride_ratio,(X,y))
        entire_train_dataset = within_subject_dataset(entire_train[0])
        entire_train_loader = torch.utils.data.DataLoader(entire_train_dataset,batch_size=BATCH_SIZE,shuffle=True)

        time_points_length = np.shape(entire_train[0][0])[2]
        number_channels = np.shape(entire_train[0][0])[1]

        checkpoint = keras.callbacks.ModelCheckpoint(
            f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//full_model_transfer_learning.keras',
            monitor='loss',
            verbose=1,
            mode='min',
            save_best_only=True)
        logger = keras.callbacks.CSVLogger(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//full_model_training.log',append=False)
        callbackslist = [checkpoint, logger]
        
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        lossfn = keras.losses.categorical_crossentropy
        model = model_config(number_channels,model_type,time_points_length,0.25)
        model.compile(loss=lossfn,optimizer=optimizer,metrics=['accuracy'])
        history = model.fit(entire_train_loader,epochs=(validation_metrics['stopped_epoch']),callbacks=callbackslist)
         
def evaluation_on_holdout(test_subject_dataset, holdout_subject,dataset_type,model_type,did_finetune, BATCH_SIZE = 32,
                                          window_series_ratio = 0.25, window_stride_ratio = 1):
    test_metrics = {'accuracy':[],'precision_score':[],'recall_score':[],
                    'f1_score':[],'confusion_matrix':[],'cohen_kappa_score':[],
                    'stopped_epoch':[]}
    
    X, y = (test_subject_dataset[0],test_subject_dataset[1])
    test = window_augment(window_series_ratio,window_stride_ratio,(X,y))
    test_dataset = within_subject_dataset(test[0])
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=False)

    if did_finetune:
        best_model = keras.saving.load_model(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//finetune//full_model_transfer_learning.keras',compile=True)
    else:
        best_model = keras.saving.load_model(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//full_model_transfer_learning.keras',compile=True)
    pred_prob_y = best_model.predict(test_loader)
    y_pred = np.argmax(pred_prob_y, axis=1)
    y_true = test_dataset.y_true()

    test_metrics = metric_calculation(test_metrics,y_true,y_pred)
    with open(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//test_metrics.pkl',"wb") as f:
                pickle.dump(test_metrics, f)

def fine_tune(pretrain_dataset,holdout_subject, lr, EPOCHS,PATIENCE,GIVE_UP, model_type,dataset_type,
            hyperparameter_tuning=True,full_training=True,
            BATCH_SIZE = 32,MIN_DELTA = 0.01,
            window_series_ratio = 0.25, window_stride_ratio = 1):
    
    X, y = (pretrain_dataset[0],pretrain_dataset[1])
    
    skf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
  
    
    if hyperparameter_tuning:
        fold_metrics = {'accuracy':[],'precision_score':[],'recall_score':[],
                    'f1_score':[],'confusion_matrix':[],'cohen_kappa_score':[],
                    'stopped_epoch':[], 'validation_loss':[], 'training_loss': []}
        finetune_metrics = fold_metrics.copy()

        for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
            os.makedirs(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//finetune//val_fold_{fold}', exist_ok=True)
        
            X_train_fold, y_train_fold = X[train_index], y[train_index]
            X_val_fold, y_val_fold = X[val_index], y[val_index]
            (train_fold,val_fold) = window_augment(window_series_ratio,window_stride_ratio,(X_train_fold,y_train_fold),(X_val_fold, y_val_fold))
            
            time_points_length = np.shape(train_fold[0])[2]
            
            train_dataset = within_subject_dataset(train_fold)
            val_dataset = within_subject_dataset(val_fold)

            train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=False)
            val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)
            
            checkpoint = keras.callbacks.ModelCheckpoint(
                f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//finetune//val_fold_{fold}//model_transfer_learning.keras',
                monitor='val_loss',
                verbose=1,
                mode='min',
                save_best_only=True,)
            earlystopping = give_up(give_up=GIVE_UP,patience=PATIENCE,min_delta=MIN_DELTA)
            logger = keras.callbacks.CSVLogger(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//finetune//val_fold_{fold}//transfer_training.log',append=False)
            callbackslist = [checkpoint, earlystopping, logger]
            
            optimizer = keras.optimizers.Adam(learning_rate=lr)
            lossfn = keras.losses.categorical_crossentropy
            model = keras.saving.load_model(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//full_model_transfer_learning.keras')

            for layer in model.layers[:-2]:
                layer.trainable = False

            model.compile(loss=lossfn,optimizer=optimizer,metrics=['accuracy'])
            history = model.fit(train_loader,epochs=EPOCHS,validation_data=val_loader,callbacks=callbackslist)

            best_model = keras.saving.load_model(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//finetune//val_fold_{fold}//model_transfer_learning.keras',compile=True)

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
                finetune_metrics[i] = np.sum(fold_metrics[i], axis=0)
            elif i == 'stopped_epoch':
                finetune_metrics[i] = int(np.mean(fold_metrics[i], axis=0))
            elif i == 'training_loss' or i == 'validation_loss':
                finetune_metrics[i] = fold_metrics[i]
            else:
                finetune_metrics[i] = (np.mean(fold_metrics[i]),np.std(fold_metrics[i]))

        with open(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//finetune_metrics.pkl',"wb") as f:
                    pickle.dump(finetune_metrics, f)

    if full_training:
        with open(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//finetune_metrics.pkl',"rb") as f:
            finetune_metrics = pickle.load(f)

        entire_train = window_augment(window_series_ratio,window_stride_ratio,(X,y))
        entire_train_dataset = within_subject_dataset(entire_train[0])
        entire_train_loader = torch.utils.data.DataLoader(entire_train_dataset,batch_size=BATCH_SIZE,shuffle=True)

        time_points_length = np.shape(entire_train[0][0])[2]

        checkpoint = keras.callbacks.ModelCheckpoint(
            f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//finetune//full_model_transfer_learning.keras',
            monitor='loss',
            verbose=1,
            mode='min',
            save_best_only=True)
        logger = keras.callbacks.CSVLogger(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//finetune//full_model_training.log',append=False)
        callbackslist = [checkpoint, logger]
        
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        lossfn = keras.losses.categorical_crossentropy
        model = model = keras.saving.load_model(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject}//full_model_transfer_learning.keras')
        for layer in model.layers[:-2]:
            layer.trainable = False
        model.compile(loss=lossfn,optimizer=optimizer,metrics=['accuracy'])
        history = model.fit(entire_train_loader,epochs=(finetune_metrics['stopped_epoch']),callbacks=callbackslist)

def run_transfer_learning(model_type,dataset_type, lr, EPOCHS,PATIENCE,GIVE_UP,MIN_DELTA, window_series_ratio, window_stride_ratio):
    """Run transfer learning training using a hold-one-out approach

    Args:
        model_type (str): Name of the nerual network model to use. Only 'EEGNet' implemented.
        dataset_type (str): Name of the dataset run training on. Has to be either 'physionet', 'BCICIV2a', or 'MindRove'
        lr (float): The learning rate of the neural network
        EPOCHS (int): The maximum number of training iterations
        PATIENCE (int): The maximum number of training iterations without a MIN_DELTA decrease in validation loss
        GIVE_UP (int): The maximum number of training iterations without the validation loss improving from the initial value
        MIN_DELTA (float): The decrease in validation loss that will reset the PATIENCE parameter
        window_series_ratio (float): The ratio between the sliding window's window length and the length of the time points of the epochs
        window_stride_ratio (float): The ratio between the sliding window's length and stride
    """
    if dataset_type == 'physionet':
        subjects = np.arange(1,10)
        loo = LeaveOneOut()
        for i, (train_index,test_index) in enumerate(loo.split(subjects)):
            train_subjects = subjects[train_index]
            holdout_subject = subjects[test_index]
            
            pre_train_dataset = physionet_Transfer_Dataset_Loader('PhysioNet',train_subjects,tmin=0,tmax=4,fmin=5,fmax=30,do_faster=True)
            transfer_learning_training(pre_train_dataset,holdout_subject[0],lr,EPOCHS,PATIENCE,GIVE_UP,model_type,dataset_type,
                                        hyperparameter_tuning=True,full_training=True,BATCH_SIZE=128,MIN_DELTA=MIN_DELTA,
                                        window_series_ratio=window_series_ratio,window_stride_ratio=window_stride_ratio)
            
            with open(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject[0]}//validation_metrics.pkl', 'rb') as f:
                validation_metrics = pickle.load(f)
                validation_metrics['subject'] = train_subjects
                metric_reporting(validation_metrics,f'checkpoint_{dataset_type}//{model_type}//transfer_learning//metric_reporting.log',
                                dataset_type, model_type,'transfer_learning',
                                is_validation=True, do_loss_vs_epoch=True)
        
            holdout_subject_dataset = physionet_Dataset_Loader('PhysioNet',subjects=holdout_subject,fmin=5,fmax=30,do_faster=True)
            holdout_subject_X = holdout_subject_dataset[holdout_subject[0]][0]
            holdout_subject_y = holdout_subject_dataset[holdout_subject[0]][1]
            holdout_subject_X_finetine, holdout_subject_X_test, holdout_subject_y_finetune, holdout_subject_y_test = train_test_split(holdout_subject_X,holdout_subject_y,
                test_size=0.2,shuffle=True, random_state=0,stratify=holdout_subject_y)
            
            fine_tune((holdout_subject_X_finetine,holdout_subject_y_finetune),holdout_subject[0],lr/10,EPOCHS,PATIENCE,GIVE_UP,model_type,dataset_type,True,True,BATCH_SIZE=32,MIN_DELTA=MIN_DELTA,
                      window_series_ratio=window_series_ratio,window_stride_ratio=window_stride_ratio)

            with open(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject[0]}//finetune_metrics.pkl', 'rb') as f:
                finetune_metrics = pickle.load(f)
                finetune_metrics['subject'] = 'PhysioNet finetune'+str(holdout_subject)
                metric_reporting(finetune_metrics, f'checkpoint_{dataset_type}//{model_type}//transfer_learning//metric_reporting.log',
                                    dataset_type, model_type,'transfer_learning',
                                    is_validation=True, do_loss_vs_epoch=True)

            evaluation_on_holdout((holdout_subject_X_test,holdout_subject_y_test),holdout_subject[0],dataset_type,model_type,True,32,window_series_ratio,window_stride_ratio)
            
            with open(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject[0]}//test_metrics.pkl', 'rb') as f:
                test_metrics = pickle.load(f)
                test_metrics['subject'] = 'PhysioNet test'+str(holdout_subject)
                metric_reporting(test_metrics,f'checkpoint_{dataset_type}//{model_type}//transfer_learning//metric_reporting.log',
                 dataset_type, model_type,'transfer_learning',
                 is_validation=False)
    
    if dataset_type == 'BCICIV2a':
        subjects = np.arange(1,10)
        loo = LeaveOneOut()
        for i, (train_index,test_index) in enumerate(loo.split(subjects)):
            train_subjects = subjects[train_index]
            holdout_subject = subjects[test_index]
            
            pre_train_dataset = BCICIV2a_Transfer_Dataset_Loader('BCICIV2a',train_subjects,tmin=0,tmax=4,fmin=5,fmax=30,do_faster=True)
            transfer_learning_training(pre_train_dataset,holdout_subject[0],lr,EPOCHS,PATIENCE,GIVE_UP,model_type,dataset_type,
                                        hyperparameter_tuning=True,full_training=True,BATCH_SIZE=128,MIN_DELTA=MIN_DELTA,
                                        window_series_ratio=window_series_ratio,window_stride_ratio=window_stride_ratio)
            
            with open(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject[0]}//validation_metrics.pkl', 'rb') as f:
                validation_metrics = pickle.load(f)
                validation_metrics['subject'] = train_subjects
                metric_reporting(validation_metrics,f'checkpoint_{dataset_type}//{model_type}//transfer_learning//metric_reporting.log',
                                dataset_type, model_type,'transfer_learning',
                                is_validation=True, do_loss_vs_epoch=True)
        
            
            holdout_subject_dataset = BCICIV2a_Dataset_Loader('BCICIV2a',subjects=holdout_subject,fmin=5,fmax=30,do_faster=True)
            holdout_subject_X = holdout_subject_dataset[holdout_subject[0]][0]
            holdout_subject_y = holdout_subject_dataset[holdout_subject[0]][1]
            holdout_subject_X_finetine, holdout_subject_X_test, holdout_subject_y_finetune, holdout_subject_y_test = train_test_split(holdout_subject_X,holdout_subject_y,
                test_size=0.2,shuffle=True, random_state=0,stratify=holdout_subject_y)
            
            fine_tune((holdout_subject_X_finetine,holdout_subject_y_finetune),holdout_subject[0],lr/10,EPOCHS,PATIENCE,GIVE_UP,model_type,dataset_type,True,True,BATCH_SIZE=32,MIN_DELTA=MIN_DELTA,
                      window_series_ratio=window_series_ratio,window_stride_ratio=window_stride_ratio)

            with open(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject[0]}//finetune_metrics.pkl', 'rb') as f:
                finetune_metrics = pickle.load(f)
                finetune_metrics['subject'] = 'BCICIV2a finetune'+str(holdout_subject)
                metric_reporting(finetune_metrics, f'checkpoint_{dataset_type}//{model_type}//transfer_learning//metric_reporting.log',
                                    dataset_type, model_type,'transfer_learning',
                                    is_validation=True, do_loss_vs_epoch=True)

            evaluation_on_holdout((holdout_subject_X_test,holdout_subject_y_test),holdout_subject[0],dataset_type,model_type,True,32,window_series_ratio,window_stride_ratio)
            
            
            with open(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject[0]}//test_metrics.pkl', 'rb') as f:
                test_metrics = pickle.load(f)
                test_metrics['subject'] = 'BCICIV2a test'+str(holdout_subject)
                metric_reporting(test_metrics,f'checkpoint_{dataset_type}//{model_type}//transfer_learning//metric_reporting.log',
                 dataset_type, model_type,'transfer_learning',
                 is_validation=False)
    
    if dataset_type == 'MindRove':
        subjects = np.arange(1,7)
        loo = LeaveOneOut()
        for i, (train_index,test_index) in enumerate(loo.split(subjects)):
            train_subjects = subjects[train_index]
            holdout_subject = subjects[test_index]
            
            pre_train_dataset = Mindrove_Transfer_Dataset_Loader('MindRove',train_subjects,tmin=0,tmax=5,fmin=5,fmax=30,load_bad_indices=True)
            transfer_learning_training(pre_train_dataset,holdout_subject[0],lr,EPOCHS,PATIENCE,GIVE_UP,model_type,dataset_type,
                                        hyperparameter_tuning=True,full_training=True,BATCH_SIZE=128,MIN_DELTA=MIN_DELTA,
                                        window_series_ratio=window_series_ratio,window_stride_ratio=window_stride_ratio)
            
            with open(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject[0]}//validation_metrics.pkl', 'rb') as f:
                validation_metrics = pickle.load(f)
                validation_metrics['subject'] = train_subjects
                metric_reporting(validation_metrics,f'checkpoint_{dataset_type}//{model_type}//transfer_learning//metric_reporting.log',
                                dataset_type, model_type,'transfer_learning',
                                is_validation=True, do_loss_vs_epoch=True)
        
        
            holdout_subject_dataset = Mindrove_Dataset_Loader('MindRove',subjects=holdout_subject,fmin=5,fmax=30,load_bad_indices=True)
            holdout_subject_X = holdout_subject_dataset[holdout_subject[0]][0]
            holdout_subject_y = holdout_subject_dataset[holdout_subject[0]][1]
            holdout_subject_X_finetine, holdout_subject_X_test, holdout_subject_y_finetune, holdout_subject_y_test = train_test_split(holdout_subject_X,holdout_subject_y,
                test_size=0.2,shuffle=True, random_state=0,stratify=holdout_subject_y)
            
            fine_tune((holdout_subject_X_finetine,holdout_subject_y_finetune),holdout_subject[0],lr/10,EPOCHS,PATIENCE,GIVE_UP,model_type,dataset_type,True,True,BATCH_SIZE=32,MIN_DELTA=MIN_DELTA,
                      window_series_ratio=window_series_ratio,window_stride_ratio=window_stride_ratio)

            with open(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject[0]}//finetune_metrics.pkl', 'rb') as f:
                finetune_metrics = pickle.load(f)
                finetune_metrics['subject'] = 'Mindrove finetune'+str(holdout_subject)
                metric_reporting(finetune_metrics, f'checkpoint_{dataset_type}//{model_type}//transfer_learning//metric_reporting.log',
                                    dataset_type, model_type,'transfer_learning',
                                    is_validation=True, do_loss_vs_epoch=True)

            evaluation_on_holdout((holdout_subject_X_test,holdout_subject_y_test),holdout_subject[0],dataset_type,model_type,True,32,window_series_ratio,window_stride_ratio)
            
            
            with open(f'checkpoint_{dataset_type}//{model_type}//transfer_learning//holdout_{holdout_subject[0]}//test_metrics.pkl', 'rb') as f:
                test_metrics = pickle.load(f)
                test_metrics['subject'] = 'Mindrove test'+str(holdout_subject)
                metric_reporting(test_metrics,f'checkpoint_{dataset_type}//{model_type}//transfer_learning//metric_reporting.log',
                 dataset_type, model_type,'transfer_learning',
                 is_validation=False)
