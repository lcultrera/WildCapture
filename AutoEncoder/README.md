This folder contains code and instructions for training and using an autoencoder (CAE) to work with attention heatmaps. The dataset used for training and testing consists of attention heatmaps extracted from the transformer model. The dataset includes heatmaps for both In-Distribution (ID) and Out-of-Distribution (OOD) sets.

## Dataset Organization

The dataset is organized in a CSV file that lists all the images (attention heatmaps).

## Dataset Structure

In this repository, we provide the attention heatmaps dataset for both In-Distribution (Used to train the Autoenceoder) and Out-of-Distribution classes (with which you can calculate the reconstruction error for OOD samples). You can find the datasets at the link: https://drive.google.com/file/d/1S9iIM33BBdagDBEqWDRTaAtFMhSreAiO/view?usp=sharing.


-inDistribution.csv: CSV file containing the path of attention heatmaps for all the images in the In-Distribution classes.
-OOD.csv: CSV file containing the path of attention heatmaps for all the images in the Out-of-Distribution classes.

The pre-trained models provided in this folder refer to the following dataset split:
In distribution: Domestic Cattle, Eurasian Badger, European Hare, Grey Wolf, Red Deer, Red Fox, Wild Boar 
Out-Of-Distribution: Beech Marten, Crested Porcupine, Domestic Dog, Domestic Horse, European Roe Deer, Persian Fallow Deer, Western Polecat, Wild Cat

However you can train from scratch the classifier and the AutoEncoder with a different split.

## Pretrained Models

You can find pretrained autoencoder weights in the folder "AutoEncoder_Weights"
