# FourierHashNet

Locality Sensitive Hashing in Fourier Frequency Domain For Soft Set Containment Search

This directory contains code necessary for running all the experiments.

## Requirements

Recent versions of Pytorch, numpy, scipy, sklearn and matplotlib are required.
Additional third party softwares used - Dr.Hash, SBERT  
You can install all the required packages using  the following command:

	$ pip install -r requirements.txt

#Datasets and trained models
Please download files from https://rebrand.ly/fhash and place in the current folder. 
This contains the original datasets, dataset splits, trained models and other intermediate data dumps for reproducing tables and plots.  


## Run Experiments

The command lines and scripts used for training models are listed commands.txt.   
Command lines specify the exact hyperparameter settings used to train the models.   

#Reproduce plots and figures  

FinalSubmission-Figs-NeurIPS23-Main.ipynb and FinalSubmission-Figs-NeurIPS23-Supp.ipynb contains code used for generating all the figures presented in the paper (main and appendix respectively).   

Notes:  
 - GPU usage is required for model training
 - Hashing is done only on CPU. 
 - source code files are all in src folder.  

