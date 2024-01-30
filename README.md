# Induction Motor Fault Detection
----
## Data - Machinery Fault Dataset - Kaggle
url: https://www.kaggle.com/datasets/uysalserkan/fault-induction-motor-dataset?resource=download

## Introduction
This project aims to create a neural network model in PyTorch to predict the status of induction motor. The goal is to predict whether the given motor is in its normal state, or imbalanced state (load value ranging from 6 gram to 35 gram)

## Data
Each CSV file contain tachometer signal, underhang and overhang bearing accelerometer, and microphone recording.

Due to the physical constraint, the 35g data is excluded due to the difference in rotation frequencies.

## Getting Started
----
The environment can be created using the `environment-macos.yml`. 

    conda env create -f environment-macos.yml
    
Windows compatible version of the yml will be added in the near future, however, the code should run on both MacOS and Windows. The neural network is optimized to utilize both Apple's MPS Machine Learning acceleration backend and Nvidia's CUDA GPU acceleration, and will automatically select the best methods.

After the environment is setup, run the `main.py` under the main dir to start training the model.