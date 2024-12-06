# Brain_Computer_Interface
## Motor Imagery EEG-based BCI 

This repository provides a program for training and evaluating the EEGNet on three motor imagery (MI) EEG-based datasets with four classes. The program supports data augmentation, artifact rejection, and both within-subject and transfer learning approaches. Comprehensive metric reporting is provided for assessing performance.

## **Features**
- Training EEGNet on 3 MI EEG-based 4-class datasets
- Supports artifact removal using the FASTER algorithm
- provides a broad signal processing pipeline 
- Data augmentation via the sliding window technique
- Has both within-subject training and transfer learning via the hold-one-out approaches
- Provides comprehensive metric reporting, including accuracy, precision, recall, F1-score, confusion matrices, and Loss against epochs graphs.


An environement.yml file is provided for creating a conda env with the necessary libraries to run the program. The EEGModels.py was adapted from [here](https://github.com/vlawhern/arl-eegmodels) and it used only the EEGNet neural network modified to work with Keras and Torch. The example_runs.py file provides examples on how to use the functions available.

## **Datasets**
The program is compatible with 2 publicly available datasets, as well as a dataset recorded for this research.

### BCI Competition Dataset 2a
The dataset can be downloaded from [here](https://www.bbci.de/competition/iv/) and it must be placed in the same directory, and the folder should be renamed to "BCICIV2a"

### PhysioNet's EEG Motor Movement/Imagery Dataset
The dataset can be downloaded from [here](https://physionet.org/content/eegmmidb/1.0.0/) and it must be placed in the same directory, and the folder should be renamed to "Physionet"

### **MindRove Arc Dataset**
The dataset can be downloaded from [here](https://github.com/Ramiz03S/MindRove_Dataset) and it must be placed in the same directory, and the folder should be renamed to "MindRove"

