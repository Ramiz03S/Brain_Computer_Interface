import os
os.environ["KERAS_BACKEND"] = "torch"
import torch
import keras
import numpy as np
import EEGModels
import logging
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator

class window():
    def __init__(self,time_series_length, window_series_ratio, window_stride_ratio):
        self.N = time_series_length
        self.w = window_series_ratio
        window_length = self.w * self.N
        self.s = window_length / window_stride_ratio

        assert (((1-self.w)*(self.N/self.s))%1 == 0), 'Windows will overhang, it will exceed the boundary of the time series'
        assert (self.w <= 1), 'Window length cannot be more than the time series length!'
        assert ((self.w > 0) and (self.N > 0) and (self.s > 0)), 'Parameters cannot be zero or negative!'
        assert (((self.w * self.N) % 1 == 0) and (self.N % 1 == 0) and (self.s % 1 == 0)), 'Parameters must be integers'
        assert (self.s <= (self.w * self.N)), 'Stride cannot be longer than window length'
    
    def generate_indices(self):
        for i in range(int((1-self.w)*(self.N/self.s)+1)):
            yield np.arange((i * self.s), (self.N * self.w) + (i * self.s),dtype=np.int64)

def window_augment(window_series_ratio, window_stride_ratio,*args):
    result = []
    for arg in args:
        data, label = arg
        n_epoch, n_channel, n_points = np.shape(data)

        window_length = window_series_ratio * n_points
        stride = window_length / window_stride_ratio

        epoch_factor = int((1-(window_length/n_points))*(n_points/stride)+1)
        augmented_data = np.zeros((n_epoch*epoch_factor, n_channel, int(window_length)))
        augmented_label = np.zeros((n_epoch*epoch_factor))

        win = window(n_points,window_series_ratio,window_stride_ratio)
        for (i, indices) in enumerate(win.generate_indices()):
            augmented_data[i*n_epoch:(i+1)*n_epoch,:,:] = data[:,:,indices]
            augmented_label[i*n_epoch:(i+1)*n_epoch] = label[:]
        
        result.append((augmented_data,augmented_label))
    return tuple(result)

class give_up(keras.callbacks.Callback):
    def __init__(self,give_up,patience,min_delta):
        super().__init__()
        self.give_up = give_up
        self.patience = patience
        self.min_delta = min_delta

    def on_train_begin(self, logs=None):
        self.keep = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.best_val_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        current_val_loss = logs.get("val_loss")
        self.stopped_epoch = epoch
        if epoch == 0:
            self.initial_val_loss = current_val_loss

        if current_val_loss > self.initial_val_loss:
                self.keep += 1
                if self.keep >= self.give_up:
                    self.stopped_epoch = epoch - self.give_up
                    self.model.stop_training = True
                    print("Restoring model weights from end of best epoch.")
                    self.model.set_weights(self.best_weights)
                    print(f"{self.give_up} epochs passed without validation loss decreasing back to the initial value")
        else :
            self.keep = 0
            if self.best_val_loss - current_val_loss > self.min_delta:
                self.best_val_loss = current_val_loss
                self.wait = 0
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch - self.patience
                    self.model.stop_training = True
                    print("Restoring model weights from end of best epoch.")
                    self.model.set_weights(self.best_weights)
                    print(f"{self.patience} epochs passed without validation loss improving")
        
class within_subject_dataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = dataset[0]
        self.labels = dataset[1]
    
    def __len__(self):
        length = np.shape(self.data)[0]
        return length
    def __getitem__(self, index):
        data = torch.from_numpy(self.data[index])
        label = torch.tensor((self.labels[index]-1),dtype=torch.long)
        label = torch.nn.functional.one_hot(label,4)
        return (data ,label)
    
    def y_true(self):
        return self.labels - 1

def model_config(number_channels, model_type, time_points_length, dropout_rate):
    
    if model_type == "EEGNet":
        model = EEGModels.EEGNet(4,number_channels,time_points_length,kernLength=32,dropoutRate=dropout_rate)

    return model

def metric_calculation(metrics, y_true, y_pred):
    # metrics = {'accuracy':[],'precision_score':[],'recall_score':[],
    #             'f1_score':[],'confusion_matrix':[],'cohen_kappa_score':[],
    #             'stopped_epoch':[], 'validation_loss':[], 'training_loss': []}
    accuracy = accuracy_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred,average='weighted')
    recall = recall_score(y_true,y_pred,average='weighted')
    f1 = f1_score(y_true,y_pred,average='weighted')
    confusion = confusion_matrix(y_true,y_pred)
    ck = cohen_kappa_score(y_true,y_pred)


    metrics['accuracy'].append(accuracy)
    metrics['precision_score'].append(precision)
    metrics['recall_score'].append(recall)
    metrics['f1_score'].append(f1)
    metrics['confusion_matrix'].append(confusion)
    metrics['cohen_kappa_score'].append(ck)

    return metrics
    
def metric_reporting(metric, log_file,dataset_type, model_type,learning_type, is_validation, do_loss_vs_epoch=False):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(filename=log_file, filemode='a',level=logging.INFO, format='%(message)s')
    if is_validation:
        precision_mean, precision_std = metric['precision_score']
        recall_mean, recall_std = metric['recall_score']
        accuracy_mean, accuracy_std = metric['accuracy']
        f1_mean, f1_std = metric['f1_score']
        confusion_matrix = metric['confusion_matrix']
        ck_mean, ck_std = metric['cohen_kappa_score']
        subject = metric['subject']

        
        logging.info(f'\nValidation metrics for subject: {subject}\n')
        logging.info(f'Accuracy: {accuracy_mean:.2f}±{accuracy_std:.2f}')
        logging.info(f'Precision: {precision_mean:.2f}±{precision_std:.2f}')
        logging.info(f'Recall: {recall_mean:.2f}±{recall_std:.2f}')
        logging.info(f'F1 Score: {f1_mean:.2f}±{f1_std:.2f}')
        logging.info(f'Cohen Kappa Score: {ck_mean:.2f}±{ck_std:.2f}')
        logging.info(f'Confusion Matrix: \n{confusion_matrix}')



        if do_loss_vs_epoch:
            validation_loss_folds = metric['validation_loss']
            training_loss_folds = metric['training_loss']
            epochs = []
            for i in range(1,6):
                epochs.append(len(validation_loss_folds[i-1]))
            max_epochs = np.max(epochs)

            plt.figure(figsize=(20,10))
            for i in range(1,6):
                epochs_plot = range(0, len(validation_loss_folds[i-1]))
                plt.plot(epochs_plot, validation_loss_folds[i-1],color=cm.Greys(0.4+0.6*i/5),
                 label = f'Validation Loss for Fold {i} of training')
            for i in range(1,6):    
                epochs_plot = range(0, len(training_loss_folds[i-1]))
                plt.plot(epochs_plot, training_loss_folds[i-1],color=cm.Blues(0.4+0.6*i/5),
                linestyle='--', label = f'Training Loss for Fold {i} of training')
                
            epochs_plot = range(0, max_epochs)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            ax = plt.gca()
            ax.yaxis.set_major_locator(MultipleLocator(0.05))
            plt.xticks(epochs_plot[::20])
            plt.legend()
            plt.title(f'Loss vs Epochs for hold-out subject {subject}')
            plt.grid(True)
            plt.savefig(f'checkpoint_{dataset_type}//{model_type}//{learning_type}//loss_vs_epochs_{subject}.png', format='png')
            plt.clf()
    
    else:
        precision = metric['precision_score']
        recall = metric['recall_score']
        accuracy = metric['accuracy']
        f1 = metric['f1_score']
        confusion_matrix = metric['confusion_matrix']
        ck = metric['cohen_kappa_score']
        subject = metric['subject']

        logging.info(f'\nTest metrics for subject: {subject}\n')
        logging.info(f'Accuracy: {accuracy[0]:.2f}')
        logging.info(f'Precision: {precision[0]:.2f}')
        logging.info(f'Recall: {recall[0]:.2f}')
        logging.info(f'F1 Score: {f1[0]:.2f}')
        logging.info(f'Cohen Kappa Score: {ck[0]:.2f}')
        logging.info(f'Confusion Matrix: \n{confusion_matrix[0]}')

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
